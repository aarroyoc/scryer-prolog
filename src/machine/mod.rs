use crate::prolog_parser_rebis::ast::*;
use crate::prolog_parser_rebis::tabled_rc::*;

use crate::clause_types::*;
use crate::forms::*;
use crate::instructions::*;
use crate::machine::heap::*;
use crate::machine::loader::*;
use crate::machine::term_stream::{LiveTermStream, LoadStatePayload, TermStream};
use crate::read::*;

mod attributed_variables;
pub(super) mod code_repo;
pub mod code_walker;
mod compile;
mod copier;
pub mod heap;
mod load_state;
mod loader;
pub mod machine_errors;
pub mod machine_indices;
pub(super) mod machine_state;
pub mod partial_string;
mod preprocessor;
mod raw_block;
mod stack;
pub(crate) mod streams;
mod term_stream;

#[macro_use]
mod arithmetic_ops;
#[macro_use]
mod machine_state_impl;
mod system_calls;

//use crate::machine::attributed_variables::*;
use crate::machine::compile::*;
use crate::machine::code_repo::*;
// use crate::machine::loader::*;
use crate::machine::machine_errors::*;
use crate::machine::machine_indices::*;
use crate::machine::machine_state::*;
use crate::machine::streams::*;

use crate::indexmap::IndexMap;

//use std::convert::TryFrom;
use std::fs::File;
use std::mem;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

#[derive(Debug)]
pub struct MachinePolicies {
    call_policy: Box<dyn CallPolicy>,
    cut_policy: Box<dyn CutPolicy>,
}

lazy_static! {
    pub static ref INTERRUPT: AtomicBool = AtomicBool::new(false);
}

impl MachinePolicies {
    #[inline]
    fn new() -> Self {
        MachinePolicies {
            call_policy: Box::new(DefaultCallPolicy {}),
            cut_policy: Box::new(DefaultCutPolicy {}),
        }
    }
}

impl Default for MachinePolicies {
    #[inline]
    fn default() -> Self {
        MachinePolicies::new()
    }
}

#[derive(Debug)]
pub(super) struct LoadContext {
    pub(super) path: PathBuf,
    pub(super) stream: Stream,
    pub(super) module: ClauseName,
}

impl LoadContext {
    #[inline]
    fn new(path: &str, stream: Stream) -> Self {
        LoadContext {
            path: PathBuf::from(path),
            stream,
            module: clause_name!("user"),
        }
    }
}

#[derive(Debug)]
pub struct Machine {
    pub(super) machine_st: MachineState,
    pub(super) inner_heap: Heap,
    pub(super) policies: MachinePolicies,
    pub(super) indices: IndexStore,
    pub(super) code_repo: CodeRepo,
    pub(super) user_input: Stream,
    pub(super) user_output: Stream,
    pub(super) load_contexts: Vec<LoadContext>,
}

#[inline]
fn current_dir() -> std::path::PathBuf {
    let mut path_buf = std::path::PathBuf::from(file!());

    path_buf.pop();
    path_buf
}

include!(concat!(env!("OUT_DIR"), "/libraries.rs"));

impl Machine {
    /*
    fn compile_special_forms(&mut self)
    {
        let verify_attrs_src = ListingSource::User;

        match compile_special_form(
            self,
            Stream::from(VERIFY_ATTRS),
            verify_attrs_src,
        )
        {
            Ok(p) => {
                self.machine_st.attr_var_init.verify_attrs_loc = p;
            }
            Err(_) =>
                panic!("Machine::compile_special_forms() failed at VERIFY_ATTRS"),
        }

        let project_attrs_src = ListingSource::User;

        match compile_special_form(
            self,
            Stream::from(PROJECT_ATTRS),
            project_attrs_src,
        )
        {
            Ok(p) => {
                self.machine_st.attr_var_init.project_attrs_loc = p;
            }
            Err(e) =>
                panic!("Machine::compile_special_forms() failed at PROJECT_ATTRS: {}", e),
        }
    }

    fn compile_scryerrc(&mut self) {
        let mut path = match dirs_next::home_dir() {
            Some(path) => path,
            None => return,
        };

        path.push(".scryerrc");

        if path.is_file() {
            let file_src = match File::open(&path) {
                Ok(file_handle) => Stream::from_file_as_input(
                    clause_name!(".scryerrc"),
                    file_handle,
                ),
                Err(_) => return,
            };

            let rc_src = ListingSource::from_file_and_path(
                clause_name!(".scryerrc"),
                path.to_path_buf(),
            );

            compile_user_module(self, file_src, rc_src);
        }
    }
*/
    #[cfg(test)]
    pub fn reset(&mut self) {
        self.current_input_stream = readline::input_stream();
        self.policies.cut_policy = Box::new(DefaultCutPolicy {});
        self.machine_st.reset();
    }

    fn run_module_predicate(&mut self, module_name: ClauseName, key: PredicateKey) {
        if let Some(module) = self.indices.modules.get(&module_name) {
            if let Some(ref code_index) = module.code_dir.get(&key) {
                let p = code_index.local().unwrap();

                self.machine_st.cp = LocalCodePtr::Halt;
                self.machine_st.p = CodePtr::Local(LocalCodePtr::DirEntry(p));

                return self.run_query();
            }
        }

        unreachable!();
    }

    fn load_file(&mut self, path: String, stream: Stream) {
        self.machine_st[temp_v!(1)] = Addr::Stream(
            self.machine_st.heap.push(HeapCellValue::Stream(
                stream,
            ))
        );

        self.machine_st[temp_v!(2)] = Addr::Con(
            self.machine_st.heap.push(HeapCellValue::Atom(
                clause_name!(path, self.machine_st.atom_tbl),
                None,
            ))
        );

        self.run_module_predicate(clause_name!("loader"), (clause_name!("file_load"), 2));
    }

    fn load_top_level(&mut self) {
        let mut path_buf = current_dir();
        path_buf.push("toplevel.pl");

        let path = path_buf.to_str().unwrap().to_string();

        self.load_file(path, Stream::from(include_str!("../toplevel.pl")));

        if let Some(toplevel) = self.indices.modules.get(&clause_name!("$toplevel")) {
            load_module(
                &mut self.indices.code_dir,
                &mut self.indices.op_dir,
                &mut self.indices.meta_predicates,
                toplevel,
            );
        } else {
            unreachable!()
        }
    }

    fn load_special_forms(&mut self) {
        let mut path_buf = current_dir();
        path_buf.push("machine/attributed_variables.pl");

        bootstrapping_compile(
            Stream::from(include_str!("attributed_variables.pl")),
            self,
            ListingSource::from_file_and_path(
                clause_name!("attributed_variables"),
                path_buf,
            ),
        ).unwrap();

        let mut path_buf = current_dir();
        path_buf.push("machine/project_attributes.pl");

        bootstrapping_compile(
            Stream::from(include_str!("project_attributes.pl")),
            self,
            ListingSource::from_file_and_path(
                clause_name!("project_attributes"),
                path_buf,
            ),
        ).unwrap();

        if let Some(module) = self.indices.modules.get(&clause_name!("$atts")) {
            if let Some(code_index) = module.code_dir.get(&(clause_name!("driver"), 2)) {
                self.machine_st.attr_var_init.verify_attrs_loc = code_index.local().unwrap();
            }
        }

        if let Some(module) = self.indices.modules.get(&clause_name!("$project_atts")) {
            if let Some(code_index) = module.code_dir.get(&(clause_name!("driver"), 2)) {
                self.machine_st.attr_var_init.project_attrs_loc = code_index.local().unwrap();
            }
        }
    }

    pub fn run_top_level(&mut self) {
        use std::env;

        let mut arg_pstrs = vec![];

        for arg in env::args() {
            arg_pstrs.push(self.machine_st.heap.put_complete_string(&arg));
        }

        let list_addr = Addr::HeapCell(self.machine_st.heap.to_list(arg_pstrs.into_iter()));

        self.machine_st[temp_v!(1)] = list_addr;

        // WAS:
        // self.run_module_predicate(clause_name!("$toplevel"), (clause_name!("$repl"), 1));

        self.run_module_predicate(clause_name!("$toplevel"), (clause_name!("repl"), 0));
    }

    pub fn new(user_input: Stream, user_output: Stream) -> Self
    {
        use crate::ref_thread_local::RefThreadLocal;

        let mut wam = Machine {
            machine_st: MachineState::new(),
            inner_heap: Heap::new(),
            policies: MachinePolicies::new(),
            indices: IndexStore::new(),
            code_repo: CodeRepo::new(),
            user_input,
            user_output,
            load_contexts: vec![],
        };

        let mut lib_path = current_dir();

        lib_path.pop();
        lib_path.push("lib");

        bootstrapping_compile(
            Stream::from(LIBRARIES.borrow()["ops_and_meta_predicates"]),
            &mut wam,
            ListingSource::from_file_and_path(
                clause_name!("ops_and_meta_predicates.pl"),
                lib_path.clone(),
            ),
        ).unwrap();

        bootstrapping_compile(
            Stream::from(LIBRARIES.borrow()["builtins"]),
            &mut wam,
            ListingSource::from_file_and_path(
                clause_name!("builtins.pl"),
                lib_path.clone(),
            ),
        ).unwrap();

        if let Some(builtins) = wam.indices.modules.get(&clause_name!("builtins")) {
            load_module(
                &mut wam.indices.code_dir,
                &mut wam.indices.op_dir,
                &mut wam.indices.meta_predicates,
                builtins,
            );
        } else {
            unreachable!()
        }

        lib_path.pop(); // remove the "lib" at the end

        bootstrapping_compile(
            Stream::from(include_str!("../term_and_goal_expansion.pl")),
            &mut wam,
            ListingSource::from_file_and_path(
                clause_name!("term_and_goal_expansion.pl"),
                lib_path.clone(),
            ),
        ).unwrap();

        bootstrapping_compile(
            Stream::from(include_str!("../loader.pl")),
            &mut wam,
            ListingSource::from_file_and_path(
                clause_name!("loader.pl"),
                lib_path.clone(),
            ),
        ).unwrap();

        if let Some(loader) = wam.indices.modules.get(&clause_name!("loader")) {
            load_module(
                &mut wam.indices.code_dir,
                &mut wam.indices.op_dir,
                &mut wam.indices.meta_predicates,
                loader,
            );
        } else {
            unreachable!()
        }

        wam.load_special_forms();
        wam.load_top_level();
        wam.configure_streams();

        wam
    }

    pub fn configure_streams(&mut self) {
        self.user_input.options.alias = Some(clause_name!("user_input"));

        self.indices.stream_aliases.insert(
            clause_name!("user_input"),
            self.user_input.clone(),
        );

        self.indices.streams.insert(
            self.user_input.clone()
        );

        self.user_output.options.alias = Some(clause_name!("user_output"));

        self.indices.stream_aliases.insert(
            clause_name!("user_output"),
            self.user_output.clone(),
        );

        self.indices.streams.insert(
            self.user_output.clone()
        );
    }

    fn throw_session_error(&mut self, err: SessionError, key: PredicateKey) {
        let h = self.machine_st.heap.h();

        let err = MachineError::session_error(h, err);
        let stub = MachineError::functor_stub(key.0, key.1);
        let err = self.machine_st.error_form(err, stub);

        self.machine_st.throw_exception(err);
        return;
    }

    fn handle_toplevel_command(&mut self, code_ptr: REPLCodePtr, p: LocalCodePtr) {
        match code_ptr {
            REPLCodePtr::AddDynamicPredicate => {
                self.add_dynamic_predicate();
            }
            REPLCodePtr::AddGoalExpansionClause => {
                self.add_goal_expansion_clause();
            }
            REPLCodePtr::AddTermExpansionClause => {
                self.add_term_expansion_clause();
            }
            REPLCodePtr::ClauseToEvacuable => {
                self.clause_to_evacuable();
            }
            REPLCodePtr::ConcludeLoad => {
                self.conclude_load();
            }
            REPLCodePtr::PopLoadContext => {
                self.pop_load_context();
            }
            REPLCodePtr::PushLoadContext => {
                self.push_load_context();
            }
            REPLCodePtr::PopLoadStatePayload => {
                self.pop_load_state_payload();
            }
            REPLCodePtr::UseModule => {
                self.use_module();
            }
            REPLCodePtr::LoadCompiledLibrary => {
                self.load_compiled_library();
            }
            REPLCodePtr::DeclareModule => {
                self.declare_module();
            }
            REPLCodePtr::PushLoadStatePayload => {
                self.push_load_state_payload();
            }
            REPLCodePtr::LoadContextSource => {
                self.load_context_source();
            }
            REPLCodePtr::LoadContextFile => {
                self.load_context_file();
            }
            REPLCodePtr::LoadContextDirectory => {
                self.load_context_directory();
            }
            REPLCodePtr::LoadContextModule => {
                self.load_context_module();
            }
            REPLCodePtr::LoadContextStream => {
                self.load_context_stream();
            }
            REPLCodePtr::MetaPredicateProperty => {
                self.meta_predicate_property();
            }
            REPLCodePtr::CompilePendingPredicates => {
                self.compile_pending_predicates();
            }
            REPLCodePtr::UserAssertz => {
                self.compile_user_assert(AppendOrPrepend::Append);
            }
            REPLCodePtr::UserAsserta => {
                self.compile_user_assert(AppendOrPrepend::Prepend);
            }
            REPLCodePtr::UserRetract => {
                self.retract_user_clause();
            }
        }

        self.machine_st.p = CodePtr::Local(p);
    }

    pub(super)
    fn run_query(&mut self) {
        while !self.machine_st.p.is_halt() {
            self.machine_st.query_stepper(
                &mut self.indices,
                &mut self.policies,
                &mut self.code_repo,
                &mut self.user_input,
                &mut self.user_output,
            );

            match self.machine_st.p {
                CodePtr::REPL(code_ptr, p) => {
                    self.handle_toplevel_command(code_ptr, p);

                    if self.machine_st.fail {
                        self.machine_st.backtrack();
                    }
                }
                /*
                CodePtr::DynamicTransaction(_trans_type, _p) => {
                    // self.code_repo.cached_query is about to be overwritten by the term expander,
                    // so hold onto it locally and restore it after the compiler has finished.
                    self.machine_st.fail = false;
                    /*
                    let cached_query = mem::replace(&mut self.code_repo.cached_query, vec![]);

                    // self.dynamic_transaction(trans_type, p);
                    self.code_repo.cached_query = cached_query;

                    if let CodePtr::Local(LocalCodePtr::TopLevel(_, 0)) = self.machine_st.p {
                        break;
                    }
                    */
                }
                */
                _ => {
                    break;
                }
            };
        }
    }
}

impl MachineState {
    fn dispatch_instr(
        &mut self,
        instr: &Line,
        indices: &mut IndexStore,
        policies: &mut MachinePolicies,
        code_repo: &CodeRepo,
        user_input: &mut Stream,
        user_output: &mut Stream,
    ) {
        match instr {
            &Line::Arithmetic(ref arith_instr) => {
                self.execute_arith_instr(arith_instr)
            }
            &Line::Choice(ref choice_instr) => {
                self.execute_choice_instr(choice_instr, &mut policies.call_policy)
            }
            &Line::Cut(ref cut_instr) => {
                self.execute_cut_instr(cut_instr, &mut policies.cut_policy)
            }
            &Line::Control(ref control_instr) => {
                self.execute_ctrl_instr(
                    indices,
                    code_repo,
                    &mut policies.call_policy,
                    &mut policies.cut_policy,
                    user_input,
                    user_output,
                    control_instr,
                )
            }
            &Line::Fact(ref fact_instr) => {
                self.execute_fact_instr(&fact_instr);
                self.p += 1;
            }
            &Line::IndexingCode(ref indexing_lines) => {
                self.execute_indexing_instr(indexing_lines, &mut policies.call_policy)
            }
            &Line::IndexedChoice(ref choice_instr) => {
                self.execute_indexed_choice_instr(choice_instr, &mut policies.call_policy)
            }
            &Line::Query(ref query_instr) => {
                self.execute_query_instr(&query_instr);
                self.p += 1;
            }
        }
    }

    fn execute_instr(
        &mut self,
        indices: &mut IndexStore,
        policies: &mut MachinePolicies,
        code_repo: &CodeRepo,
        user_input: &mut Stream,
        user_output: &mut Stream,
    ) {
        let instr = match code_repo.lookup_instr(self.last_call, &self.p) {
            Some(instr) => instr,
            None => return,
        };

        self.dispatch_instr(
            instr.as_ref(),
            indices,
            policies,
            code_repo,
            user_input,
            user_output,
        );
    }

    fn backtrack(&mut self) {
        // if self.b > 0 {
        let b = self.b;

        self.b0 = self.stack.index_or_frame(b).prelude.b0;
        self.p = CodePtr::Local(self.stack.index_or_frame(b).prelude.bp);

        /*
        if let CodePtr::Local(LocalCodePtr::TopLevel(_, p)) = self.p {
            self.fail = p == 0;
        } else {
        */
        self.fail = false;
        // }
        /*} else {
            self.p = CodePtr::Local(LocalCodePtr::TopLevel(0, 0));
        }*/
    }

    fn check_machine_index(&mut self, code_repo: &CodeRepo) -> bool {
        match self.p {
            CodePtr::Local(LocalCodePtr::DirEntry(p)) |
            CodePtr::Local(LocalCodePtr::IndexingBuf(p, ..))
                if p < code_repo.code.len() => {
                }
            CodePtr::Local(LocalCodePtr::Halt) | CodePtr::REPL(..) => {
                return false;
            }
            /*
            CodePtr::DynamicTransaction(..) => {
                // prevent use of dynamic transactions from
                // succeeding in expansions. self.fail will be toggled
                // back to false later.
                self.fail = true;
                return false;
            }
            */
            _ => {
            }
        }

        true
    }

    // return true iff verify_attr_interrupt is called.
    fn verify_attr_stepper(
        &mut self,
        indices: &mut IndexStore,
        policies: &mut MachinePolicies,
        code_repo: &mut CodeRepo,
        user_input: &mut Stream,
        user_output: &mut Stream,
    ) -> bool {
        loop {
            let instr = match code_repo.lookup_instr(self.last_call, &self.p) {
                Some(instr) => {
                    if instr.as_ref().is_head_instr() {
                        instr
                    } else {
                        let cp = self.p.local();
                        self.run_verify_attr_interrupt(cp);
                        return true;
                    }
                }
                None => return false,
            };

            self.dispatch_instr(
                instr.as_ref(),
                indices,
                policies,
                code_repo,
                user_input,
                user_output,
            );

            if self.fail {
                self.backtrack();
            }

            if !self.check_machine_index(code_repo) {
                return false;
            }
        }
    }

    fn run_verify_attr_interrupt(&mut self, cp: LocalCodePtr) {
        let p = self.attr_var_init.verify_attrs_loc;

        self.attr_var_init.cp = cp;
        self.verify_attr_interrupt(p);
    }

    fn query_stepper(
        &mut self,
        indices: &mut IndexStore,
        policies: &mut MachinePolicies,
        code_repo: &mut CodeRepo,
        user_input: &mut Stream,
        user_output: &mut Stream,
    ) {
        loop {
            self.execute_instr(
                indices,
                policies,
                code_repo,
                user_input,
                user_output,
            );

            if self.fail {
                self.backtrack();
            }

            match self.p {
                CodePtr::VerifyAttrInterrupt(_) => {
                    self.p = CodePtr::Local(self.attr_var_init.cp);

                    let instigating_p = CodePtr::Local(self.attr_var_init.instigating_p);
                    let instigating_instr = code_repo.lookup_instr(false, &instigating_p).unwrap();

                    if !instigating_instr.as_ref().is_head_instr() {
                        let cp = self.p.local();
                        self.run_verify_attr_interrupt(cp);
                    } else if !self.verify_attr_stepper(
                        indices,
                        policies,
                        code_repo,
                        user_input,
                        user_output,
                    ) {
                        if self.fail {
                            break;
                        }

                        let cp = self.p.local();
                        self.run_verify_attr_interrupt(cp);
                    }
                }
                _ => {
                    if !self.check_machine_index(code_repo) {
                        break;
                    }
                }
            }
        }
    }
}
