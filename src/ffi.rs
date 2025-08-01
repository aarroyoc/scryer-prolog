/* How does FFI work?

Each WAM machine has a ForeignFunctionTable instance that contains a table of functions and structs.

Structs are defined via foreign_struct/2. Basic types are defined by libffi, but struct types need to
be manually defined to get an ffi_type. Additionally, to recover structs from return arguments, we store
fields and atom_fields, as a way to lookup the content of the struct (fields) and the nested structs (atom_fields).

Functions are defined via use_foreign_module/2. It opens a library and leaks the memory of the library,
to prevent Rust freeing the memory. There's no way to recover that memory at the moment. We get a pointer for
each function and we build a CIF for each one, with the input arguments and the return argument.

Exec happens via '$foreign_call', we find the function, we try to cast the values that we have to the definition
of the function, we reserve memory for them and we build an array of pointers. To get the return argument, we
reserve enough memory for the return and we build the Scryer values from them.

Structs are a bit tricky as they need to be aligned. For that, we reserve enough memory (libffi calculates that)
and for each field: we add to the pointer until we're aligned to the next data type we're going to write, we write it,
and finally we add the pointer the size of what we've written.
*/

use crate::arena::Arena;
use crate::atom_table::Atom;
use crate::forms::Number;
use crate::parser::ast::Fixnum;

use dashu::Integer;
use ordered_float::OrderedFloat;
use std::alloc::{self, Layout};
use std::any::Any;
use std::collections::HashMap;
use std::error::Error;
use std::ffi::{c_void, CString};
use std::ptr::addr_of_mut;

use libffi::low::type_tag::STRUCT;
use libffi::low::{ffi_abi_FFI_DEFAULT_ABI, ffi_cif, ffi_type, prep_cif, types, CodePtr};
use libloading::{Library, Symbol};

pub struct FunctionDefinition {
    pub name: String,
    pub return_value: Atom,
    pub args: Vec<Atom>,
}

#[derive(Debug)]
pub struct FunctionImpl {
    cif: ffi_cif,
    args: Vec<*mut ffi_type>,
    code_ptr: CodePtr,
    return_struct_name: Option<String>,
}

#[derive(Debug, Default)]
pub struct ForeignFunctionTable {
    table: HashMap<String, FunctionImpl>,
    structs: HashMap<String, StructImpl>,
}

#[derive(Clone)]
struct StructImpl {
    ffi_type: ffi_type,
    fields: Vec<*mut ffi_type>,
    atom_fields: Vec<Atom>,
}

impl std::fmt::Debug for StructImpl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StructImpl")
            .field("ffi_type", &&"<???>")
            .field("fields", &self.fields)
            .field("atom_fields", &self.atom_fields)
            .finish()
    }
}

struct PointerArgs {
    pointers: Vec<*mut c_void>,
    _memory: Vec<Box<dyn Any>>,
}

impl ForeignFunctionTable {
    pub fn merge(&mut self, other: ForeignFunctionTable) {
        self.table.extend(other.table);
    }

    pub fn define_struct(&mut self, name: &str, atom_fields: Vec<Atom>) -> Result<(), FFIError> {
        let mut fields: Vec<_> = atom_fields
            .iter()
            .map(|x| self.map_type_ffi(x))
            .collect::<Result<_, FFIError>>()?;
        fields.push(std::ptr::null_mut::<ffi_type>());
        let struct_type = ffi_type {
            type_: STRUCT,
            elements: fields.as_mut_ptr(),
            ..Default::default()
        };
        self.structs.insert(
            name.to_string(),
            StructImpl {
                ffi_type: struct_type,
                fields,
                atom_fields,
            },
        );
        Ok(())
    }

    fn map_type_ffi(&mut self, source: &Atom) -> Result<*mut ffi_type, FFIError> {
        Ok(match source {
            atom!("sint64") => addr_of_mut!(types::sint64),
            atom!("sint32") => addr_of_mut!(types::sint32),
            atom!("sint16") => addr_of_mut!(types::sint16),
            atom!("sint8") => addr_of_mut!(types::sint8),
            atom!("uint64") => addr_of_mut!(types::uint64),
            atom!("uint32") => addr_of_mut!(types::uint32),
            atom!("uint16") => addr_of_mut!(types::uint16),
            atom!("uint8") => addr_of_mut!(types::uint8),
            atom!("bool") => addr_of_mut!(types::sint8),
            atom!("void") => addr_of_mut!(types::void),
            atom!("cstr") => addr_of_mut!(types::pointer),
            atom!("ptr") => addr_of_mut!(types::pointer),
            atom!("f32") => addr_of_mut!(types::float),
            atom!("f64") => addr_of_mut!(types::double),
            struct_name => match self.structs.get_mut(&*struct_name.as_str()) {
                Some(ref mut struct_type) => &mut struct_type.ffi_type,
                None => return Err(FFIError::InvalidFFIType),
            },
        })
    }

    pub(crate) fn load_library(
        &mut self,
        library_name: &str,
        functions: &Vec<FunctionDefinition>,
    ) -> Result<(), Box<dyn Error>> {
        let mut ff_table: ForeignFunctionTable = Default::default();
        unsafe {
            let library = Library::new(library_name)?;
            for function in functions {
                let symbol_name: CString = CString::new(function.name.clone())?;
                let code_ptr: Symbol<*mut c_void> =
                    library.get(&symbol_name.into_bytes_with_nul())?;
                let mut args: Vec<_> = function
                    .args
                    .iter()
                    .map(|x| self.map_type_ffi(x))
                    .collect::<Result<_, FFIError>>()?;
                let mut cif: ffi_cif = Default::default();
                prep_cif(
                    &mut cif,
                    ffi_abi_FFI_DEFAULT_ABI,
                    args.len(),
                    self.map_type_ffi(&function.return_value)?,
                    args.as_mut_ptr(),
                )
                .unwrap();

                let return_struct_name = if (*self.map_type_ffi(&function.return_value)?).type_
                    as u32
                    == libffi::raw::FFI_TYPE_STRUCT
                {
                    Some(function.return_value.as_str().to_string())
                } else {
                    None
                };

                ff_table.table.insert(
                    function.name.clone(),
                    FunctionImpl {
                        cif,
                        args,
                        code_ptr: CodePtr(code_ptr.into_raw().as_raw_ptr()),
                        return_struct_name,
                    },
                );
            }
            std::mem::forget(library);
        }
        self.merge(ff_table);
        Ok(())
    }

    fn build_pointer_args(
        args: &mut [Value],
        type_args: &[*mut ffi_type],
        structs_table: &mut HashMap<String, StructImpl>,
    ) -> Result<PointerArgs, FFIError> {
        let mut pointers = Vec::with_capacity(args.len());
        let mut _memory = Vec::new();
        for i in 0..args.len() {
            let field_type = type_args[i];
            unsafe {
                macro_rules! push_int {
                    ($type:ty) => {{
                        let n: $type = args[i].as_int()?;
                        let mut box_value = Box::new(n) as Box<dyn Any>;
                        pointers.push(&mut *box_value as *mut _ as *mut c_void);
                        _memory.push(box_value);
                    }};
                }

                match (*field_type).type_ as u32 {
                    libffi::raw::FFI_TYPE_UINT8 => push_int!(u8),
                    libffi::raw::FFI_TYPE_SINT8 => push_int!(i8),
                    libffi::raw::FFI_TYPE_UINT16 => push_int!(u16),
                    libffi::raw::FFI_TYPE_SINT16 => push_int!(i16),
                    libffi::raw::FFI_TYPE_UINT32 => push_int!(u32),
                    libffi::raw::FFI_TYPE_SINT32 => push_int!(i32),
                    libffi::raw::FFI_TYPE_UINT64 => push_int!(u64),
                    libffi::raw::FFI_TYPE_SINT64 => push_int!(i64),
                    libffi::raw::FFI_TYPE_FLOAT => {
                        let n: f32 = args[i].as_float()? as f32;
                        let mut box_value = Box::new(n) as Box<dyn Any>;
                        pointers.push(&mut *box_value as *mut _ as *mut c_void);
                        _memory.push(box_value);
                    }
                    libffi::raw::FFI_TYPE_DOUBLE => {
                        let n: f64 = args[i].as_float()?;
                        let mut box_value = Box::new(n) as Box<dyn Any>;
                        pointers.push(&mut *box_value as *mut _ as *mut c_void);
                        _memory.push(box_value);
                    }
                    libffi::raw::FFI_TYPE_POINTER => {
                        let ptr: *mut c_void = args[i].as_ptr()?;
                        pointers.push(ptr);
                    }
                    libffi::raw::FFI_TYPE_STRUCT => {
                        let (mut ptr, _size, _align) =
                            Self::build_struct(&mut args[i], structs_table)?;
                        pointers.push(&mut *ptr as *mut _ as *mut c_void);
                        _memory.push(ptr);
                    }
                    _ => return Err(FFIError::InvalidFFIType),
                }
            }
        }
        Ok(PointerArgs { pointers, _memory })
    }

    fn build_struct(
        arg: &mut Value,
        structs_table: &mut HashMap<String, StructImpl>,
    ) -> Result<(Box<dyn Any>, usize, usize), FFIError> {
        match arg {
            Value::Struct(ref name, ref mut struct_args) => {
                if let Some(ref mut struct_type) = structs_table.clone().get_mut(name) {
                    let layout = Layout::from_size_align(
                        struct_type.ffi_type.size,
                        struct_type.ffi_type.alignment.into(),
                    )
                    .unwrap();
                    let align = struct_type.ffi_type.alignment as usize;
                    let size = struct_type.ffi_type.size;
                    let ptr = unsafe { alloc::alloc(layout) as *mut c_void };

                    if ptr.is_null() {
                        panic!("allocation failed")
                    }

                    let mut field_ptr = ptr;

                    #[allow(clippy::needless_range_loop)]
                    for i in 0..(struct_type.fields.len() - 1) {
                        macro_rules! try_write_int {
                            ($type:ty) => {{
                                field_ptr = field_ptr
                                    .add(field_ptr.align_offset(std::mem::align_of::<$type>()));
                                let n: $type = struct_args[i].as_int()?;
                                std::ptr::write(field_ptr as *mut $type, n);
                                field_ptr = field_ptr.add(std::mem::size_of::<$type>());
                            }};
                        }

                        macro_rules! write {
                            ($type:ty, $value:expr) => {{
                                let data: $type = $value;
                                std::ptr::write(field_ptr as *mut $type, data);
                                field_ptr = field_ptr.add(align);
                            }};
                        }

                        let field = struct_type.fields[i];
                        unsafe {
                            match (*field).type_ as u32 {
                                libffi::raw::FFI_TYPE_UINT8 => try_write_int!(u8),
                                libffi::raw::FFI_TYPE_SINT8 => try_write_int!(i8),
                                libffi::raw::FFI_TYPE_UINT16 => try_write_int!(u16),
                                libffi::raw::FFI_TYPE_SINT16 => try_write_int!(i16),
                                libffi::raw::FFI_TYPE_UINT32 => try_write_int!(u32),
                                libffi::raw::FFI_TYPE_SINT32 => try_write_int!(i32),
                                libffi::raw::FFI_TYPE_UINT64 => try_write_int!(u64),
                                libffi::raw::FFI_TYPE_SINT64 => try_write_int!(i64),
                                libffi::raw::FFI_TYPE_POINTER => {
                                    write!(*mut c_void, struct_args[i].as_ptr()?)
                                }
                                libffi::raw::FFI_TYPE_FLOAT => {
                                    write!(f32, struct_args[i].as_float()? as f32)
                                }
                                libffi::raw::FFI_TYPE_DOUBLE => {
                                    write!(f64, struct_args[i].as_float()?)
                                }
                                libffi::raw::FFI_TYPE_STRUCT => {
                                    let (struct_ptr, struct_size, struct_align) =
                                        Self::build_struct(&mut struct_args[i], structs_table)?;
                                    field_ptr = field_ptr.add(field_ptr.align_offset(struct_align));

                                    std::ptr::copy(
                                        &*struct_ptr as *const _ as *const c_void,
                                        field_ptr,
                                        struct_size,
                                    );
                                    field_ptr = field_ptr.add(struct_size);
                                }
                                _ => {
                                    unreachable!()
                                }
                            }
                        }
                    }

                    #[allow(clippy::from_raw_with_void_ptr)]
                    Ok((unsafe { Box::from_raw(ptr) }, size, align))
                } else {
                    Err(FFIError::InvalidStructName)
                }
            }
            _ => Err(FFIError::ValueCast),
        }
    }

    pub fn exec(
        &mut self,
        name: &str,
        mut args: Vec<Value>,
        arena: &mut Arena,
    ) -> Result<Value, FFIError> {
        let function_impl = self.table.get_mut(name).ok_or(FFIError::FunctionNotFound)?;
        let mut pointer_args =
            Self::build_pointer_args(&mut args, &function_impl.args, &mut self.structs)?;

        unsafe {
            macro_rules! call_and_return {
                ($type:ty) => {{
                    let mut n: $type = 0;
                    libffi::raw::ffi_call(
                        &mut function_impl.cif,
                        Some(*function_impl.code_ptr.as_safe_fun()),
                        &mut n as *mut _ as *mut c_void,
                        pointer_args.pointers.as_mut_ptr() as *mut *mut c_void,
                    );
                    Ok(Value::Number(fixnum!(Number, n, arena)))
                }};
            }

            match (*function_impl.cif.rtype).type_ as u32 {
                libffi::raw::FFI_TYPE_VOID => call_and_return!(i32),
                libffi::raw::FFI_TYPE_UINT8 => call_and_return!(u8),
                libffi::raw::FFI_TYPE_SINT8 => call_and_return!(i8),
                libffi::raw::FFI_TYPE_UINT16 => call_and_return!(u16),
                libffi::raw::FFI_TYPE_SINT16 => call_and_return!(i16),
                libffi::raw::FFI_TYPE_UINT32 => call_and_return!(u32),
                libffi::raw::FFI_TYPE_SINT32 => call_and_return!(i32),
                libffi::raw::FFI_TYPE_UINT64 => call_and_return!(u64),
                libffi::raw::FFI_TYPE_SINT64 => call_and_return!(i64),
                libffi::raw::FFI_TYPE_POINTER => {
                    let mut n: *mut c_void = std::ptr::null_mut();
                    libffi::raw::ffi_call(
                        &mut function_impl.cif,
                        Some(*function_impl.code_ptr.as_safe_fun()),
                        &mut n as *mut *mut c_void as *mut c_void,
                        pointer_args.pointers.as_mut_ptr(),
                    );
                    Ok(Value::Number(fixnum!(Number, n as isize, arena)))
                }
                libffi::raw::FFI_TYPE_FLOAT => {
                    let mut n: f32 = 0.0;
                    libffi::raw::ffi_call(
                        &mut function_impl.cif,
                        Some(*function_impl.code_ptr.as_safe_fun()),
                        &mut n as *mut _ as *mut c_void,
                        pointer_args.pointers.as_mut_ptr(),
                    );
                    Ok(Value::Number(Number::Float(OrderedFloat(n.into()))))
                }
                libffi::raw::FFI_TYPE_DOUBLE => {
                    let mut n: f64 = 0.0;
                    libffi::raw::ffi_call(
                        &mut function_impl.cif,
                        Some(*function_impl.code_ptr.as_safe_fun()),
                        &mut n as *mut _ as *mut c_void,
                        pointer_args.pointers.as_mut_ptr(),
                    );
                    Ok(Value::Number(Number::Float(OrderedFloat(n))))
                }
                libffi::raw::FFI_TYPE_STRUCT => {
                    let name = &function_impl
                        .return_struct_name
                        .clone()
                        .ok_or(FFIError::StructNotFound)?;
                    let struct_type = self.structs.get(name).ok_or(FFIError::StructNotFound)?;
                    let layout = Layout::from_size_align(
                        struct_type.ffi_type.size,
                        struct_type.ffi_type.alignment.into(),
                    )
                    .unwrap();
                    let ptr = alloc::alloc(layout) as *mut c_void;

                    if ptr.is_null() {
                        panic!("allocation failed")
                    }

                    libffi::raw::ffi_call(
                        &mut function_impl.cif,
                        Some(*function_impl.code_ptr.as_safe_fun()),
                        &mut *ptr as *mut _,
                        pointer_args.pointers.as_mut_ptr(),
                    );
                    let struct_val = self.read_struct(ptr, name, struct_type, arena);
                    #[allow(clippy::from_raw_with_void_ptr)]
                    drop(Box::from_raw(ptr));
                    struct_val
                }
                _ => unreachable!(),
            }
        }
    }

    fn read_struct(
        &self,
        ptr: *mut c_void,
        name: &str,
        struct_type: &StructImpl,
        arena: &mut Arena,
    ) -> Result<Value, FFIError> {
        unsafe {
            let mut returns = Vec::new();
            let mut field_ptr = ptr;

            for i in 0..(struct_type.fields.len() - 1) {
                let field = struct_type.fields[i];

                macro_rules! read_and_push_int {
                    ($type:ty) => {{
                        field_ptr =
                            field_ptr.add(field_ptr.align_offset(std::mem::align_of::<$type>()));
                        let n = std::ptr::read(field_ptr as *mut $type);
                        returns.push(Value::Number(fixnum!(Number, n, arena)));
                        field_ptr = field_ptr.add(std::mem::size_of::<$type>());
                    }};
                }

                match (*field).type_ as u32 {
                    libffi::raw::FFI_TYPE_UINT8 => read_and_push_int!(u8),
                    libffi::raw::FFI_TYPE_SINT8 => read_and_push_int!(i8),
                    libffi::raw::FFI_TYPE_UINT16 => read_and_push_int!(u16),
                    libffi::raw::FFI_TYPE_SINT16 => read_and_push_int!(i16),
                    libffi::raw::FFI_TYPE_UINT32 => read_and_push_int!(u32),
                    libffi::raw::FFI_TYPE_SINT32 => read_and_push_int!(i32),
                    libffi::raw::FFI_TYPE_UINT64 => read_and_push_int!(u64),
                    libffi::raw::FFI_TYPE_SINT64 => read_and_push_int!(i64),
                    libffi::raw::FFI_TYPE_POINTER => read_and_push_int!(i64),
                    libffi::raw::FFI_TYPE_FLOAT => {
                        field_ptr =
                            field_ptr.add(field_ptr.align_offset(std::mem::align_of::<f32>()));
                        let n: f32 = std::ptr::read(field_ptr as *mut f32);
                        returns.push(Value::Number(Number::Float(OrderedFloat(n.into()))));
                        field_ptr = field_ptr.add(std::mem::size_of::<f32>());
                    }
                    libffi::raw::FFI_TYPE_DOUBLE => {
                        field_ptr =
                            field_ptr.add(field_ptr.align_offset(std::mem::align_of::<f64>()));
                        let n: f64 = std::ptr::read(field_ptr as *mut f64);
                        returns.push(Value::Number(Number::Float(OrderedFloat(n))));
                        field_ptr = field_ptr.add(std::mem::size_of::<f64>());
                    }
                    libffi::raw::FFI_TYPE_STRUCT => {
                        let substruct = struct_type.atom_fields[i].as_str();
                        let struct_type = self
                            .structs
                            .get(&*substruct)
                            .ok_or(FFIError::StructNotFound)?;
                        field_ptr = field_ptr
                            .add(field_ptr.align_offset(struct_type.ffi_type.alignment as usize));
                        let struct_val =
                            self.read_struct(field_ptr, &substruct, struct_type, arena);
                        returns.push(struct_val?);
                        field_ptr = field_ptr.add(struct_type.ffi_type.size);
                    }
                    _ => {
                        unreachable!()
                    }
                }
            }
            Ok(Value::Struct(name.into(), returns))
        }
    }
}

#[derive(Clone, Debug)]
pub enum Value {
    Number(Number),
    CString(CString),
    Struct(String, Vec<Value>),
}

impl Value {
    fn as_int<I>(&self) -> Result<I, FFIError>
    where
        Integer: TryInto<I>,
        i64: TryInto<I>,
    {
        match self {
            Value::Number(Number::Integer(ibig_ptr)) => {
                let ibig: &Integer = ibig_ptr;
                ibig.clone().try_into().map_err(|_| FFIError::ValueDontFit)
            }
            Value::Number(Number::Fixnum(fixnum)) => fixnum
                .get_num()
                .try_into()
                .map_err(|_| FFIError::ValueDontFit),
            _ => Err(FFIError::ValueCast),
        }
    }

    fn as_float(&self) -> Result<f64, FFIError> {
        match self {
            &Value::Number(Number::Float(OrderedFloat(f))) => Ok(f),
            _ => Err(FFIError::ValueCast),
        }
    }

    fn as_ptr(&mut self) -> Result<*mut c_void, FFIError> {
        match self {
            Value::CString(ref mut cstr) => Ok(&mut *cstr as *mut _ as *mut c_void),
            Value::Number(Number::Fixnum(fixnum)) => Ok(std::ptr::with_exposed_provenance_mut(
                fixnum.get_num() as usize,
            )),
            _ => Err(FFIError::ValueCast),
        }
    }
}

#[derive(Debug)]
pub enum FFIError {
    ValueCast,
    ValueDontFit,
    InvalidFFIType,
    InvalidStructName,
    FunctionNotFound,
    StructNotFound,
}

impl std::fmt::Display for FFIError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
}

impl Error for FFIError {}
