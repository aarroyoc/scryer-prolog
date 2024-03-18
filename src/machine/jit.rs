use std::collections::HashMap;

use crate::instructions::*;
use crate::machine::*;

use cranelift::prelude::*;
use cranelift::prelude::codegen::ir::immediates::Offset32;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

struct CompileOutput {
    module: JITModule,
    code_ptr: *const u8,
}

#[derive(Debug, PartialEq)]
pub enum JitCompileError {
    UndefinedPredicate,
    InstructionNotImplemented,
}

pub struct JitMachine {
    modules: HashMap<String, CompileOutput>,
    machine_state: *const u8,
    registers: *const u8,
    write_literal_to_var: *const u8,
}

impl std::fmt::Debug for JitMachine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JitMachine")
    }
}

impl JitMachine {
    pub fn new(machine_st: &MachineState) -> Self {

	JitMachine {
	    modules: HashMap::new(),
	    machine_state: machine_st as *const MachineState as *const u8,
	    registers: machine_st.registers.as_ptr() as *const u8,
	    write_literal_to_var: MachineState::write_literal_to_var as *const u8,
	}
    }

    // For now, one module = one predicate
    // Functions must take N parameters (arity)
    // Access to MachineState via global pointer
    // MachineState Registers + ShadowRegisters??
    // Use TAIL call convention
    pub fn compile(&mut self, name: &str, code: Code) -> Result<(), JitCompileError>{
        let mut builder = JITBuilder::with_flags(&[("preserve_frame_pointers", "true")], cranelift_module::default_libcall_names()).unwrap();
	builder.symbols(self.modules.iter().map(|(k, v)| (k,v.code_ptr)));
	
	let mut module = JITModule::new(builder);
	let mut ctx = module.make_context();
	let mut func_ctx = FunctionBuilderContext::new();

	let mut sig = module.make_signature();
	// Set arguments/returns
	sig.call_conv = isa::CallConv::Tail;
	ctx.func.signature = sig.clone();

	let mut func = module.declare_function(name, Linkage::Local, &sig).unwrap();

	let mut fn_builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
	let block = fn_builder.create_block();
	fn_builder.switch_to_block(block);
	for wam_instr in code {
	    match wam_instr {
		Instruction::Proceed => {
		    fn_builder.ins().return_(&[]);
		    fn_builder.seal_all_blocks();
		    fn_builder.finalize();
		    break;
		}
		Instruction::ExecuteNamed(arity, pred_name, ..) => {
		    let mut callee_func_sig = module.make_signature();
		    callee_func_sig.call_conv = isa::CallConv::Tail;
		    // right now, all predicates have 0 arguments. In the future with shadow registers, we could improve this
		    if let Ok(callee_func) = module.declare_function(&format!("{}/{}", pred_name.as_str(), arity), Linkage::Import, &callee_func_sig) {
			let func_ref = module.declare_func_in_func(callee_func, fn_builder.func);
			fn_builder.ins().return_call(func_ref, &[]);
			fn_builder.seal_all_blocks();
			fn_builder.finalize();
			break;
		    } else {
			return Err(JitCompileError::UndefinedPredicate);
		    }
		}
		Instruction::GetConstant(_, c, reg) => {
		    let reg_ptr = fn_builder.ins().iconst(types::I64, self.registers as i64); // TODO: call deref
		    let reg_num = reg.reg_num();
		    let reg_value = fn_builder.ins().load(types::I64, MemFlags::new(), reg_ptr, Offset32::new((reg_num as i32)*8));
		    let c = unsafe { std::mem::transmute::<u64, i64>(u64::from(c)) };		    
		    let c_value = fn_builder.ins().iconst(types::I64, c);
		    let machine_state_value = fn_builder.ins().iconst(types::I64, self.machine_state as i64);
		    let mut sig = module.make_signature();
		    sig.call_conv = isa::CallConv::SystemV;
		    sig.params.push(AbiParam::new(types::I64));
		    sig.params.push(AbiParam::new(types::I64));
		    sig.params.push(AbiParam::new(types::I64));
		    let sig_ref = fn_builder.import_signature(sig);
		    let write_literal_to_var = fn_builder.ins().iconst(types::I64, self.write_literal_to_var as i64);
		    fn_builder.ins().call_indirect(sig_ref, write_literal_to_var, &[machine_state_value, reg_value, c_value]);
		}
		Instruction::PutConstant(_, c, reg) => {
		    let reg_ptr = fn_builder.ins().iconst(types::I64, self.registers as i64);
		    let reg_num = reg.reg_num();
		    let c = unsafe { std::mem::transmute::<u64, i64>(u64::from(c)) };
		    let c_value = fn_builder.ins().iconst(types::I64, c);
		    fn_builder.ins().store(MemFlags::new(), c_value, reg_ptr, Offset32::new((reg_num as i32)*8));
		}
		_ => {
		    return Err(JitCompileError::InstructionNotImplemented);
		}
	    }
	}
	module.define_function(func, &mut ctx).unwrap();
	module.clear_context(&mut ctx);

	module.finalize_definitions().unwrap();

	let code_ptr = unsafe { std::mem::transmute(module.get_finalized_function(func)) };
	self.modules.insert(name.to_string(), CompileOutput {
	    module,
	    code_ptr
	});

	Ok(())
    }

    pub fn exec(&self, name: &str) -> Result<(), ()> {
	if let Some(output) = self.modules.get(name) {
	    let code_ptr = unsafe { std::mem::transmute::<_, extern "C" fn() -> ()>(output.code_ptr) };
	    code_ptr();
	    Ok(())
	} else {
	    Err(())
	}
    }
}

// basic.
#[test]
fn jit_test_proceed() {
    let machine_st = MachineState::new();
    let code = vec![Instruction::Proceed];
    let name = "basic/0";

    let mut jit = JitMachine::new(&machine_st);
    assert_eq!(jit.compile(name, code), Ok(()));
    jit.exec(name);
}

// basic.
// simple :- basic.
#[test]
fn jit_test_execute_named() {
    let machine_st = MachineState::new();
    let mut jit = JitMachine::new(&machine_st);
    let code = vec![Instruction::Proceed];
    let name = "basic/0";
    assert_eq!(jit.compile(name, code), Ok(()));

    let code = vec![Instruction::ExecuteNamed(0, atom!("basic"), CodeIndex::default(&mut Arena::new()))];
    let name = "simple/0";
    assert_eq!(jit.compile(name, code), Ok(()));
    jit.exec(name);
}

// a(5).
// b :- a(5).
#[test]
fn jit_test_get_constant() {
    let machine_st = MachineState::new();
    let mut jit = JitMachine::new(&machine_st);
    let code = vec![Instruction::GetConstant(Level::Shallow, fixnum_as_cell!(Fixnum::build_with(5)), RegType::Temp(1)), Instruction::Proceed];
    let name = "a/1";
    assert_eq!(jit.compile(name, code), Ok(()));

    let code = vec![Instruction::PutConstant(Level::Shallow, fixnum_as_cell!(Fixnum::build_with(5)), RegType::Temp(1)), Instruction::ExecuteNamed(1, atom!("a"), CodeIndex::default(&mut Arena::new()))];
    let name = "b/0";
    assert_eq!(jit.compile(name, code), Ok(()));
    jit.exec(name);
    assert_eq!(machine_st.fail, false);
}

// a(5).
// b :- a(6).
#[test]
fn jit_test_get_constant_fail() {
    let machine_st = MachineState::new();
    let machine_st = Box::pin(MachineState::new());
    let mut jit = JitMachine::new(&machine_st);
    let code = vec![Instruction::GetConstant(Level::Shallow, fixnum_as_cell!(Fixnum::build_with(5)), RegType::Temp(1)), Instruction::Proceed];
    let name = "a/1";
    assert_eq!(jit.compile(name, code), Ok(()));

    let code = vec![Instruction::PutConstant(Level::Shallow, fixnum_as_cell!(Fixnum::build_with(6)), RegType::Temp(1)), Instruction::ExecuteNamed(1, atom!("a"), CodeIndex::default(&mut Arena::new()))];
    let name = "b/0";
    assert_eq!(jit.compile(name, code), Ok(()));
    jit.exec(name);
    assert_eq!(machine_st.fail, true);
}
