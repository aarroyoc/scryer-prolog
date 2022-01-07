use crate::arena::*;
use crate::atom_table::*;
use crate::types::*;
use crate::forms::*;
use crate::heap_iter::*;
use crate::machine::attributed_variables::*;
use crate::machine::copier::*;
use crate::machine::heap::*;
use crate::machine::machine_errors::*;
use crate::machine::machine_indices::*;
use crate::machine::machine_state::*;
use crate::machine::partial_string::*;
use crate::machine::stack::*;
use crate::parser::ast::*;
use crate::parser::rug::{Integer, Rational};

use ordered_float::*;

use indexmap::IndexSet;

use std::cmp::Ordering;
use std::convert::TryFrom;

impl MachineState {
    pub(crate) fn new() -> Self {
        MachineState {
            arena: Arena::new(),
            atom_tbl: AtomTable::new(),
            pdl: Vec::with_capacity(1024),
            s: HeapPtr::default(),
            p: 0,
            oip: 0,
            iip: 0,
            b: 0,
            b0: 0,
            e: 0,
            num_of_args: 0,
            cp: 0,
            attr_var_init: AttrVarInitializer::new(0),
            fail: false,
            heap: Heap::with_capacity(256 * 256),
            mode: MachineMode::Write,
            stack: Stack::new(),
            registers: [heap_loc_as_cell!(0); MAX_ARITY + 1], // self.registers[0] is never used.
            trail: vec![],
            tr: 0,
            hb: 0,
            block: 0,
            ball: Ball::new(),
            lifted_heap: Heap::new(),
            interms: vec![Number::default();256],
            cont_pts: Vec::with_capacity(256),
            cwil: CWIL::new(),
            flags: MachineFlags::default(),
            cc: 0,
            global_clock: 0,
            dynamic_mode: FirstOrNext::First,
            unify_fn: MachineState::unify,
            bind_fn: MachineState::bind,
            run_cleaners_fn: |_| { false },
            increment_call_count_fn: |_| { Ok(()) },
        }
    }

    #[inline]
    pub(crate) fn store(&self, value: HeapCellValue) -> HeapCellValue {
        read_heap_cell!(value,
            (HeapCellValueTag::AttrVar | HeapCellValueTag::Var, h) => {
                self.heap[h]
            }
            (HeapCellValueTag::StackVar, s) => {
                self.stack[s]
            }
            _ => {
                value
            }
        )
    }

    pub fn deref(&self, mut addr: HeapCellValue) -> HeapCellValue {
        loop {
            let value = self.store(addr);

            if value.is_var() && value != addr {
                addr = value;
                continue;
            }

            return addr;
        }
    }

    pub fn trail(&mut self, r: TrailRef) {
        match r {
            TrailRef::Ref(r) => {
                let h = r.get_value() as usize;

                match r.get_tag() {
                    RefTag::HeapCell => {
                        if h < self.hb {
                            self.trail.push(TrailEntry::build_with(
                                TrailEntryTag::TrailedHeapVar,
                                h as u64,
                            ));

                            self.tr += 1;
                        }
                    }
                    RefTag::StackCell => {
                        if h < self.b {
                            self.trail.push(TrailEntry::build_with(
                                TrailEntryTag::TrailedStackVar,
                                h as u64,
                            ));

                            self.tr += 1;
                        }
                    }
                    RefTag::AttrVar => {
                        if h < self.hb {
                            self.trail.push(TrailEntry::build_with(
                                TrailEntryTag::TrailedAttrVar,
                                h as u64,
                            ));

                            self.tr += 1;
                        }
                    }
                }
            }
            TrailRef::AttrVarHeapLink(h) => {
                if h < self.hb {
                    self.trail.push(TrailEntry::build_with(
                        TrailEntryTag::TrailedAttrVarHeapLink,
                        h as u64,
                    ));

                    self.tr += 1;
                }
            }
            TrailRef::AttrVarListLink(h, l) => {
                if h < self.hb {
                    self.trail.push(TrailEntry::build_with(
                        TrailEntryTag::TrailedAttrVarListLink,
                        h as u64,
                    ));

                    self.trail.push(TrailEntry::from_bytes(
                        list_loc_as_cell!(l).into_bytes()
                    ));

                    self.tr += 2;
                }
            }
            TrailRef::BlackboardEntry(key_atom) => {
                self.trail.push(TrailEntry::build_with(
                    TrailEntryTag::TrailedBlackboardEntry,
                    key_atom.index as u64,
                ));

                self.tr += 1;
            }
            TrailRef::BlackboardOffset(key_atom, value_cell) => {
                self.trail.push(TrailEntry::build_with(
                    TrailEntryTag::TrailedBlackboardOffset,
                    key_atom.index as u64,
                ));

                self.trail.push(TrailEntry::from_bytes(
                    value_cell.into_bytes(),
                ));

                self.tr += 2;
            }
        }
    }

    pub fn allocate(&mut self, num_cells: usize) {
        let e = self.stack.allocate_and_frame(num_cells);
        let and_frame = self.stack.index_and_frame_mut(e);

        and_frame.prelude.e = self.e;
        and_frame.prelude.cp = self.cp;

        self.e = e;
        self.p += 1;
    }

    pub fn bind(&mut self, r1: Ref, a2: HeapCellValue) {
        let t1 = self.store(r1.as_heap_cell_value());
        let t2 = self.store(a2);

        if t1.is_var() && (!t2.is_var() || a2 < r1) {
            match r1.get_tag() {
                RefTag::StackCell => {
                    self.stack[r1.get_value() as usize] = t2;
                    self.trail(TrailRef::Ref(r1));
                }
                RefTag::HeapCell => {
                    self.heap[r1.get_value() as usize] = t2;
                    self.trail(TrailRef::Ref(r1));
                }
                RefTag::AttrVar => {
                    self.bind_attr_var(r1.get_value() as usize, t2);
                }
            };
        } else {
            read_heap_cell!(a2,
                (HeapCellValueTag::StackVar, s) => {
                    self.stack[s] = t1;
                    self.trail(TrailRef::Ref(Ref::stack_cell(s)));
                }
                (HeapCellValueTag::Var, h) => {
                    self.heap[h] = t1;
                    self.trail(TrailRef::Ref(Ref::heap_cell(h)));
                }
                (HeapCellValueTag::AttrVar, h) => {
                    self.bind_attr_var(h, t1);
                }
                _ => {
                    unreachable!();
                }
            );
        }
    }

    pub fn bind_attr_var(&mut self, h: usize, addr: HeapCellValue) {
        read_heap_cell!(addr,
            (HeapCellValueTag::Var, hc) => {
                self.heap[hc] = attr_var_as_cell!(h);
                self.trail(TrailRef::Ref(Ref::heap_cell(hc)));
            }
            (HeapCellValueTag::StackVar, hc) => {
                self.stack[hc] = attr_var_as_cell!(h);
                self.trail(TrailRef::Ref(Ref::stack_cell(hc)));
            }
            _ => {
                self.push_attr_var_binding(h, addr);
                self.heap[h] = addr;
                self.trail(TrailRef::Ref(Ref::attr_var(h)));
            }
        )
    }

    fn unify_structure(&mut self, s1: usize, value: HeapCellValue) {
        // s1 is the value of a STR cell.
        let (n1, a1) = cell_as_atom_cell!(self.heap[s1]).get_name_and_arity();

        read_heap_cell!(value,
            (HeapCellValueTag::Str, s2) => {
                let (n2, a2) = cell_as_atom_cell!(self.heap[s2])
                    .get_name_and_arity();

                if n1 == n2 && a1 == a2 {
                    for idx in 0..a1 {
                        self.pdl.push(heap_loc_as_cell!(s2+1+idx));
                        self.pdl.push(heap_loc_as_cell!(s1+1+idx));
                    }
                } else {
                    self.fail = true;
                }
            }
            (HeapCellValueTag::Lis, l2) => {
                if a1 == 2 && n1 == atom!(".") {
                    for idx in 0..2 {
                        self.pdl.push(heap_loc_as_cell!(l2+1+idx));
                        self.pdl.push(heap_loc_as_cell!(s1+1+idx));
                    }
                } else {
                    self.fail = true;
                }
            }
            (HeapCellValueTag::Atom, (n2, a2)) => {
                if !(a1 == 0 && a2 == 0 && n1 == n2) {
                    self.fail = true;
                }
            }
            (HeapCellValueTag::AttrVar, h) => {
                self.bind(Ref::attr_var(h), str_loc_as_cell!(s1));
            }
            (HeapCellValueTag::Var, h) => {
                self.bind(Ref::heap_cell(h), str_loc_as_cell!(s1));
            }
            (HeapCellValueTag::StackVar, s) => {
                self.bind(Ref::stack_cell(s), str_loc_as_cell!(s1));
            }
            _ => {
                self.fail = true;
            }
        )
    }

    fn unify_list(&mut self, l1: usize, d2: HeapCellValue) {
        read_heap_cell!(d2,
            (HeapCellValueTag::Lis, l2) => {
                for idx in 0..2 {
                    self.pdl.push(heap_loc_as_cell!(l2 + idx));
                    self.pdl.push(heap_loc_as_cell!(l1 + idx));
                }
            }
            (HeapCellValueTag::Str, s2) => {
                let (n2, a2) = cell_as_atom_cell!(self.heap[s2])
                    .get_name_and_arity();

                if a2 == 2 && n2 == atom!(".") {
                    for idx in 0..2 {
                        self.pdl.push(heap_loc_as_cell!(s2+1+idx));
                        self.pdl.push(heap_loc_as_cell!(l1+idx));
                    }
                } else {
                    self.fail = true;
                }
            }
            (HeapCellValueTag::PStrLoc | HeapCellValueTag::CStr | HeapCellValueTag::PStr) => {
                self.unify_partial_string(list_loc_as_cell!(l1), d2)
            }
            (HeapCellValueTag::AttrVar, h) => {
                self.bind(Ref::attr_var(h), list_loc_as_cell!(l1));
            }
            (HeapCellValueTag::Var, h) => {
                self.bind(Ref::heap_cell(h), list_loc_as_cell!(l1));
            }
            (HeapCellValueTag::StackVar, s) => {
                self.bind(Ref::stack_cell(s), list_loc_as_cell!(l1));
            }
            _ => {
                self.fail = true;
            }
        )
    }

    pub fn unify_complete_string(&mut self, atom: Atom, value: HeapCellValue) {
        if let Some(r) = value.as_var() {
            self.bind(r, atom_as_cstr_cell!(atom));
            return;
        }

        read_heap_cell!(value,
            (HeapCellValueTag::CStr, cstr_atom) => {
                self.fail = atom != cstr_atom;
            }
            (HeapCellValueTag::Str | HeapCellValueTag::Lis | HeapCellValueTag::PStrLoc) => {
                self.unify_partial_string(atom_as_cstr_cell!(atom), value);

                if !self.pdl.is_empty() {
                    self.unify();
                }
            }
            _ => {
                self.fail = true;
            }
        );
    }

    // d1's tag is LIS, STR or PSTRLOC.
    pub fn unify_partial_string(&mut self, d1: HeapCellValue, d2: HeapCellValue) {
        if let Some(r) = d2.as_var() {
            self.bind(r, d1);
            return;
        }

        let s1 = self.heap.len();

        self.heap.push(d1);
        self.heap.push(d2);

        let mut pstr_iter1 = HeapPStrIter::new(&self.heap, s1);
        let mut pstr_iter2 = HeapPStrIter::new(&self.heap, s1 + 1);

        match compare_pstr_prefixes(&mut pstr_iter1, &mut pstr_iter2) {
            PStrCmpResult::Ordered(Ordering::Equal) => {}
            PStrCmpResult::Ordered(Ordering::Less) => {
                if pstr_iter2.focus.as_var().is_none() {
                    self.fail = true;
                } else {
                    self.pdl.push(empty_list_as_cell!());
                    self.pdl.push(pstr_iter2.focus);
                }
            }
            PStrCmpResult::Ordered(Ordering::Greater) => {
                if pstr_iter1.focus.as_var().is_none() {
                    self.fail = true;
                } else {
                    self.pdl.push(empty_list_as_cell!());
                    self.pdl.push(pstr_iter1.focus);
                }
            }
            continuable @ PStrCmpResult::FirstIterContinuable(iteratee) |
            continuable @ PStrCmpResult::SecondIterContinuable(iteratee) => {
                if continuable.is_second_iter() {
                    std::mem::swap(&mut pstr_iter1, &mut pstr_iter2);
                }

                let mut chars_iter = PStrCharsIter {
                    iter: pstr_iter1,
                    item: Some(iteratee),
                };

                let mut focus = pstr_iter2.focus;

                'outer: loop {
                    while let Some(c) = chars_iter.peek() {
                        read_heap_cell!(focus,
                            (HeapCellValueTag::Lis, l) => {
                                let val = pstr_iter2.heap[l];

                                self.pdl.push(val);
                                self.pdl.push(char_as_cell!(c));

                                focus = pstr_iter2.heap[l+1];
                            }
                            (HeapCellValueTag::Str, s) => {
                                let (name, arity) = cell_as_atom_cell!(pstr_iter2.heap[s])
                                    .get_name_and_arity();

                                if name == atom!(".") && arity == 2 {
                                    self.pdl.push(pstr_iter2.heap[s+1]);
                                    self.pdl.push(char_as_cell!(c));

                                    focus = pstr_iter2.heap[s+2];
                                } else {
                                    self.fail = true;
                                    break 'outer;
                                }
                            }
                            (HeapCellValueTag::AttrVar | HeapCellValueTag::Var, h) => {
                                match chars_iter.item.unwrap() {
                                    PStrIteratee::Char(focus, _) => {
                                        self.pdl.push(self.heap[focus]);
                                        self.pdl.push(heap_loc_as_cell!(h));
                                    }
                                    PStrIteratee::PStrSegment(focus, _, n) => {
                                        read_heap_cell!(self.heap[focus],
                                            (HeapCellValueTag::CStr | HeapCellValueTag::PStr, pstr_atom) => {
                                                if focus < self.heap.len() - 2 {
                                                    self.heap.pop();
                                                    self.heap.pop();
                                                }

                                                if n == 0 {
                                                    let target_cell = match self.heap[focus].get_tag() {
                                                        HeapCellValueTag::CStr => {
                                                            atom_as_cstr_cell!(pstr_atom)
                                                        }
                                                        HeapCellValueTag::PStr => {
                                                            pstr_loc_as_cell!(focus)
                                                        }
                                                        _ => {
                                                            unreachable!()
                                                        }
                                                    };

                                                    self.pdl.push(target_cell);
                                                    self.pdl.push(heap_loc_as_cell!(h));
                                                } else {
                                                    let h_len = self.heap.len();

                                                    self.heap.push(pstr_offset_as_cell!(focus));
                                                    self.heap.push(fixnum_as_cell!(
                                                        Fixnum::build_with(n as i64)
                                                    ));

                                                    self.pdl.push(pstr_loc_as_cell!(h_len));
                                                    self.pdl.push(heap_loc_as_cell!(h));
                                                }

                                                return;
                                            }
                                            (HeapCellValueTag::PStrOffset, pstr_loc) => {
                                                let n0 = cell_as_fixnum!(self.heap[focus+1])
                                                    .get_num() as usize;

                                                if pstr_loc < self.heap.len() - 2 {
                                                    self.heap.pop();
                                                    self.heap.pop();
                                                }

                                                if n == n0 {
                                                    self.pdl.push(pstr_loc_as_cell!(focus));
                                                    self.pdl.push(heap_loc_as_cell!(h));
                                                } else {
                                                    let h_len = self.heap.len();

                                                    self.heap.push(pstr_offset_as_cell!(pstr_loc));
                                                    self.heap.push(fixnum_as_cell!(
                                                        Fixnum::build_with(n as i64)
                                                    ));

                                                    self.pdl.push(pstr_loc_as_cell!(h_len));
                                                    self.pdl.push(heap_loc_as_cell!(h));
                                                }

                                                return;
                                            }
                                            _ => {
                                            }
                                        );

                                        if focus < self.heap.len() - 2 {
                                            self.heap.pop();
                                            self.heap.pop();
                                        }

                                        self.pdl.push(self.heap[focus]);
                                        self.pdl.push(heap_loc_as_cell!(h));

                                        return;
                                    }
                                }

                                break 'outer;
                            }
                            _ => {
                                self.fail = true;
                                break 'outer;
                            }
                        );

                        chars_iter.next();
                    }

                    chars_iter.iter.next();

                    self.pdl.push(chars_iter.iter.focus);
                    self.pdl.push(focus);

                    break;
                }
            }
            PStrCmpResult::Unordered => {
                self.pdl.push(pstr_iter1.focus);
                self.pdl.push(pstr_iter2.focus);
            }
        }

        self.heap.pop();
        self.heap.pop();
    }

    pub fn unify_atom(&mut self, atom: Atom, value: HeapCellValue) {
        read_heap_cell!(value,
            (HeapCellValueTag::Atom, (name, arity)) => {
                self.fail = !(arity == 0 && name == atom);
            }
            (HeapCellValueTag::Char, c1) => {
                if let Some(c2) = atom.as_char() {
                    self.fail = c1 != c2;
                } else {
                    self.fail = true;
                }
            }
            (HeapCellValueTag::AttrVar, h) => {
                self.bind(Ref::attr_var(h), atom_as_cell!(atom));
            }
            (HeapCellValueTag::Var, h) => {
                self.bind(Ref::heap_cell(h), atom_as_cell!(atom));
            }
            (HeapCellValueTag::StackVar, s) => {
                self.bind(Ref::stack_cell(s), atom_as_cell!(atom));
            }
            _ => {
                self.fail = true;
            }
        );
    }

    pub fn unify_char(&mut self, c: char, value: HeapCellValue) {
        read_heap_cell!(value,
            (HeapCellValueTag::Atom, (name, arity)) => {
                if let Some(c2) = name.as_char() {
                    self.fail = !(c == c2 && arity == 0);
                } else {
                    self.fail = true;
                }
            }
            (HeapCellValueTag::Char, c2) => {
                if c != c2 {
                    self.fail = true;
                }
            }
            (HeapCellValueTag::AttrVar, h) => {
                self.bind(Ref::attr_var(h), char_as_cell!(c));
            }
            (HeapCellValueTag::Var, h) => {
                self.bind(Ref::heap_cell(h), char_as_cell!(c));
            }
            (HeapCellValueTag::StackVar, s) => {
                self.bind(Ref::stack_cell(s), char_as_cell!(c));
            }
            _ => {
                self.fail = true;
            }
        );
    }

    pub fn unify_fixnum(&mut self, n1: Fixnum, value: HeapCellValue) {
        if let Some(r) = value.as_var() {
            self.bind(r, fixnum_as_cell!(n1));
            return;
        }

        match Number::try_from(value) {
            Ok(n2) => match n2 {
                Number::Fixnum(n2) if n1.get_num() == n2.get_num() => {}
                Number::Integer(n2) if n1.get_num() == *n2 => {}
                Number::Rational(n2) if n1.get_num() == *n2 => {}
                _ => {
                    self.fail = true;
                }
            },
            Err(_) => {
                self.fail = true;
            }
        }
    }

    pub fn unify_big_int(&mut self, n1: TypedArenaPtr<Integer>, value: HeapCellValue) {
        if let Some(r) = value.as_var() {
            self.bind(r, typed_arena_ptr_as_cell!(n1));
            return;
        }

        match Number::try_from(value) {
            Ok(n2) => match n2 {
                Number::Fixnum(n2) if *n1 == n2.get_num() => {}
                Number::Integer(n2) if *n1 == *n2 => {}
                Number::Rational(n2) if *n1 == *n2 => {}
                _ => {
                    self.fail = true;
                }
            },
            Err(_) => {
                self.fail = true;
            }
        }
    }

    pub fn unify_rational(&mut self, n1: TypedArenaPtr<Rational>, value: HeapCellValue) {
        if let Some(r) = value.as_var() {
            self.bind(r, typed_arena_ptr_as_cell!(n1));
            return;
        }

        match Number::try_from(value) {
            Ok(n2) => match n2 {
                Number::Fixnum(n2) if *n1 == n2.get_num() => {}
                Number::Integer(n2) if *n1 == *n2 => {}
                Number::Rational(n2) if *n1 == *n2 => {}
                _ => {
                    self.fail = true;
                }
            },
            Err(_) => {
                self.fail = true;
            }
        }
    }

    pub fn unify_f64(&mut self, f1: F64Ptr, value: HeapCellValue) {
        if let Some(r) = value.as_var() {
            self.bind(r, typed_arena_ptr_as_cell!(f1));
            return;
        }

        read_heap_cell!(value,
            (HeapCellValueTag::F64, f2) => {
                self.fail = **f1 != **f2;
            }
            (HeapCellValueTag::Cons, cons_ptr) => {
                match_untyped_arena_ptr!(cons_ptr,
                     (ArenaHeaderTag::F64, f2) => {
                         self.fail = **f1 != **F64Ptr(f2);
                     }
                     _ => {
                         self.fail = true;
                     }
                );
            }
            _ => {
                self.fail = true;
            }
        );
    }

    pub fn unify_constant(&mut self, ptr: UntypedArenaPtr, value: HeapCellValue) {
        if let Some(ptr2) = value.to_untyped_arena_ptr() {
            if ptr.get_ptr() == ptr2.get_ptr() {
                return;
            }
        }

        match_untyped_arena_ptr!(ptr,
             (ArenaHeaderTag::Integer, int_ptr) => {
                 self.unify_big_int(int_ptr, value);
             }
             (ArenaHeaderTag::Rational, rat_ptr) => {
                 self.unify_rational(rat_ptr, value);
             }
             _ => {
                 if let Some(r) = value.as_var() {
                     self.bind(r, untyped_arena_ptr_as_cell!(ptr));
                 } else {
                     self.fail = true;
                 }
             }
        );
    }

    pub fn unify(&mut self) {
        let mut tabu_list: IndexSet<(usize, usize)> = IndexSet::new();
        // self.fail = false;

        while !(self.pdl.is_empty() || self.fail) {
            let s1 = self.pdl.pop().unwrap();
            let s1 = self.deref(s1);

            let s2 = self.pdl.pop().unwrap();
            let s2 = self.deref(s2);

            if s1 != s2 {
                let d1 = self.store(s1);
                let d2 = self.store(s2);

                read_heap_cell!(d1,
                    (HeapCellValueTag::AttrVar, h) => {
                        self.bind(Ref::attr_var(h), d2);
                    }
                    (HeapCellValueTag::Var, h) => {
                        self.bind(Ref::heap_cell(h), d2);
                    }
                    (HeapCellValueTag::StackVar, s) => {
                        self.bind(Ref::stack_cell(s), d2);
                    }
                    (HeapCellValueTag::Atom, (name, arity)) => {
                        debug_assert!(arity == 0);
                        self.unify_atom(name, d2);
                    }
                    (HeapCellValueTag::Str, s1) => {
                        if d2.is_constant() {
                            self.fail = true;
                            break;
                        }

                        let s2 = s2.get_value() as usize;

                        if tabu_list.contains(&(s1, s2)) {
                            continue;
                        }

                        self.unify_structure(s1, d2);

                        if !self.fail {
                            tabu_list.insert((s1, s2));
                        }
                    }
                    (HeapCellValueTag::Lis, l1) => {
                        if d2.is_ref() {
                            let l2 = s2.get_value();

                            if tabu_list.contains(&(l1, l2)) {
                                continue;
                            }

                            tabu_list.insert((l1, l2));
                        }

                        self.unify_list(l1, d2);
                    }
                    (HeapCellValueTag::PStrLoc, pstr1_loc) => {
                        read_heap_cell!(d2,
                            (HeapCellValueTag::PStrLoc |
                             HeapCellValueTag::Lis |
                             HeapCellValueTag::Str,
                             pstr2_loc) => {
                                if tabu_list.contains(&(pstr1_loc, pstr2_loc)) {
                                    continue;
                                }
                            }
                            (HeapCellValueTag::CStr |
                             HeapCellValueTag::AttrVar |
                             HeapCellValueTag::Var |
                             HeapCellValueTag::StackVar) => {
                            }
                            _ => {
                                self.fail = true;
                                break;
                            }
                        );

                        self.unify_partial_string(d1, d2);

                        if !self.fail && !d2.is_constant() {
                            tabu_list.insert((pstr1_loc, d2.get_value()));
                        }
                    }
                    (HeapCellValueTag::CStr) => {
                        read_heap_cell!(d2,
                            (HeapCellValueTag::AttrVar, h) => {
                                self.bind(Ref::attr_var(h), d1);
                                continue;
                            }
                            (HeapCellValueTag::Var, h) => {
                                self.bind(Ref::heap_cell(h), d1);
                                continue;
                            }
                            (HeapCellValueTag::StackVar, s) => {
                                self.bind(Ref::stack_cell(s), d1);
                                continue;
                            }
                            (HeapCellValueTag::Str |
                             HeapCellValueTag::Lis |
                             HeapCellValueTag::PStrLoc) => {
                            }
                            (HeapCellValueTag::CStr) => {
                                self.fail = d1 != d2;
                                continue;
                            }
                            _ => {
                                self.fail = true;
                                return;
                            }
                        );

                        self.unify_partial_string(d2, d1);
                    }
                    (HeapCellValueTag::F64, f1) => {
                        self.unify_f64(f1, d2);
                    }
                    (HeapCellValueTag::Fixnum, n1) => {
                        self.unify_fixnum(n1, d2);
                    }
                    (HeapCellValueTag::Char, c1) => {
                        self.unify_char(c1, d2);
                    }
                    (HeapCellValueTag::Cons, ptr_1) => {
                        self.unify_constant(ptr_1, d2);
                    }
                    _ => {
                        unreachable!();
                    }
                );
            }
        }
    }

    pub(super) fn set_ball(&mut self) {
        self.ball.reset();

        let addr = self.registers[1];
        self.ball.boundary = self.heap.len();

        copy_term(
            CopyBallTerm::new(&mut self.stack, &mut self.heap, &mut self.ball.stub),
            addr,
            AttrVarPolicy::DeepCopy,
        );
    }

    pub(super) fn unwind_stack(&mut self) {
        self.b = self.block;
        self.fail = true;
    }

    #[inline]
    pub fn bind_with_occurs_check(&mut self, r: Ref, value: HeapCellValue) -> bool {
        if let RefTag::StackCell = r.get_tag() {
            // local variable optimization -- r cannot occur in the
            // heap structure bound to value, so don't bother
            // traversing value.
            self.bind(r, value);
            return false;
        }

        let mut occurs_triggered = false;

        if !value.is_constant() {
            for addr in stackful_preorder_iter(&mut self.heap, value) {
                let addr = unmark_cell_bits!(addr);

                if let Some(inner_r) = addr.as_var() {
                    if r == inner_r {
                        occurs_triggered = true;
                        break;
                    }
                }
            }
        }

        if occurs_triggered {
            self.fail = true;
        } else {
            self.bind(r, value);
        }

        return occurs_triggered;
    }

    #[inline]
    pub(super) fn bind_with_occurs_check_wrapper(&mut self, r: Ref, value: HeapCellValue) {
        self.bind_with_occurs_check(r, value);
    }

    #[inline]
    pub(super) fn bind_with_occurs_check_with_error_wrapper(
        &mut self,
        r: Ref,
        value: HeapCellValue,
    ) {
        if self.bind_with_occurs_check(r, value) {
            let err = self.representation_error(RepFlag::Term);
            let stub = functor_stub(atom!("unify_with_occurs_check"), 2);
            let err = self.error_form(err, stub);

            self.throw_exception(err);
        }
    }

    pub(super) fn unify_with_occurs_check_with_error(&mut self) {
        let mut throw_error = false;
        self.unify_with_occurs_check_loop(|| throw_error = true);

        if throw_error {
            let err = self.representation_error(RepFlag::Term);
            let stub = functor_stub(atom!("unify_with_occurs_check"), 2);
            let err = self.error_form(err, stub);

            self.throw_exception(err);
        }
    }

    pub(super) fn unify_with_occurs_check(&mut self) {
        self.unify_with_occurs_check_loop(|| {})
    }

    fn unify_structure_with_occurs_check(
        &mut self,
        s1: usize,
        value: HeapCellValue,
        mut occurs_trigger: impl FnMut(),
    ) {
        // s1 is the value of a STR cell.
        let (n1, a1) = cell_as_atom_cell!(self.heap[s1]).get_name_and_arity();

        read_heap_cell!(value,
            (HeapCellValueTag::Str, s2) => {
                let (n2, a2) = cell_as_atom_cell!(self.heap[s2])
                    .get_name_and_arity();

                if n1 == n2 && a1 == a2 {
                    for idx in 0..a1 {
                        self.pdl.push(heap_loc_as_cell!(s2+1+idx));
                        self.pdl.push(heap_loc_as_cell!(s1+1+idx));
                    }
                } else {
                    self.fail = true;
                }
            }
            (HeapCellValueTag::Lis, l2) => {
                if a1 == 2 && n1 == atom!(".") {
                    for idx in 0..2 {
                        self.pdl.push(heap_loc_as_cell!(l2+idx));
                        self.pdl.push(heap_loc_as_cell!(s1+1+idx));
                    }
                } else {
                    self.fail = true;
                }
            }
            (HeapCellValueTag::Atom, (n2, a2)) => {
                if !(a1 == 0 && a2 == 0 && n1 == n2) {
                    self.fail = true;
                }
            }
            (HeapCellValueTag::AttrVar, h) => {
                if self.bind_with_occurs_check(Ref::attr_var(h), str_loc_as_cell!(s1)) {
                    occurs_trigger();
                }
            }
            (HeapCellValueTag::Var, h) => {
                if self.bind_with_occurs_check(Ref::heap_cell(h), str_loc_as_cell!(s1)) {
                    occurs_trigger();
                }
            }
            (HeapCellValueTag::StackVar, s) => {
                if self.bind_with_occurs_check(Ref::stack_cell(s), str_loc_as_cell!(s1)) {
                    occurs_trigger();
                }
            }
            _ => {
                self.fail = true;
            }
        )
    }

    // the return value of unify_partial_string_with_occurs_check is
    // interpreted as follows:
    //
    // Some(None) -- the strings are equal, nothing to unify
    // Some(Some(f2,f1)) -- prefixes equal, try to unify focus values f2, f1
    // None -- prefixes not equal, unification fails
    //
    // d1's tag is assumed to be one of LIS, STR or PSTRLOC.
    pub fn unify_partial_string_with_occurs_check(
        &mut self,
        d1: HeapCellValue,
        d2: HeapCellValue,
        mut occurs_trigger: impl FnMut(),
    ) {
        if let Some(r) = d2.as_var() {
            if self.bind_with_occurs_check(r, d1) {
                occurs_trigger();
            }

            return;
        }

        let s1 = self.heap.len();

        self.heap.push(d1);
        self.heap.push(d2);

        let mut pstr_iter1 = HeapPStrIter::new(&self.heap, s1);
        let mut pstr_iter2 = HeapPStrIter::new(&self.heap, s1 + 1);

        match compare_pstr_prefixes(&mut pstr_iter1, &mut pstr_iter2) {
            PStrCmpResult::Ordered(Ordering::Equal) => {}
            PStrCmpResult::Ordered(Ordering::Less) => {
                if pstr_iter2.focus.as_var().is_none() {
                    self.fail = true;
                } else {
                    self.pdl.push(empty_list_as_cell!());
                    self.pdl.push(pstr_iter2.focus);
                }
            }
            PStrCmpResult::Ordered(Ordering::Greater) => {
                if pstr_iter1.focus.as_var().is_none() {
                    self.fail = true;
                } else {
                    self.pdl.push(empty_list_as_cell!());
                    self.pdl.push(pstr_iter1.focus);
                }
            }
            continuable @ PStrCmpResult::FirstIterContinuable(iteratee) |
            continuable @ PStrCmpResult::SecondIterContinuable(iteratee) => {
                if continuable.is_second_iter() {
                    std::mem::swap(&mut pstr_iter1, &mut pstr_iter2);
                }

                let mut chars_iter = PStrCharsIter {
                    iter: pstr_iter1,
                    item: Some(iteratee),
                };

                let mut focus = pstr_iter2.focus;

                'outer: loop {
                    while let Some(c) = chars_iter.peek() {
                        read_heap_cell!(focus,
                            (HeapCellValueTag::Lis, l) => {
                                let val = pstr_iter2.heap[l];

                                self.pdl.push(val);
                                self.pdl.push(char_as_cell!(c));

                                focus = pstr_iter2.heap[l+1];
                            }
                            (HeapCellValueTag::Str, s) => {
                                let (name, arity) = cell_as_atom_cell!(pstr_iter2.heap[s])
                                    .get_name_and_arity();

                                if name == atom!(".") && arity == 2 {
                                    self.pdl.push(pstr_iter2.heap[s+1]);
                                    self.pdl.push(char_as_cell!(c));

                                    focus = pstr_iter2.heap[s+2];
                                } else {
                                    self.fail = true;
                                    break 'outer;
                                }
                            }
                            (HeapCellValueTag::AttrVar | HeapCellValueTag::Var, h) => {
                                match chars_iter.item.unwrap() {
                                    PStrIteratee::Char(focus, _) => {
                                        self.pdl.push(self.heap[focus]);
                                        self.pdl.push(heap_loc_as_cell!(h));
                                    }
                                    PStrIteratee::PStrSegment(focus, _, n) => {
                                        read_heap_cell!(self.heap[focus],
                                            (HeapCellValueTag::CStr | HeapCellValueTag::PStr, pstr_atom) => {
                                                if focus < self.heap.len() - 2 {
                                                    self.heap.pop();
                                                    self.heap.pop();
                                                }

                                                if n == 0 {
                                                    let target_cell = match self.heap[focus].get_tag() {
                                                        HeapCellValueTag::CStr => {
                                                            atom_as_cstr_cell!(pstr_atom)
                                                        }
                                                        HeapCellValueTag::PStr => {
                                                            pstr_loc_as_cell!(focus)
                                                        }
                                                        _ => {
                                                            unreachable!()
                                                        }
                                                    };

                                                    self.pdl.push(target_cell);
                                                    self.pdl.push(heap_loc_as_cell!(h));
                                                } else {
                                                    let h_len = self.heap.len();

                                                    self.heap.push(pstr_offset_as_cell!(focus));
                                                    self.heap.push(fixnum_as_cell!(
                                                        Fixnum::build_with(n as i64)
                                                    ));

                                                    self.pdl.push(pstr_loc_as_cell!(h_len));
                                                    self.pdl.push(heap_loc_as_cell!(h));
                                                }

                                                return;
                                            }
                                            (HeapCellValueTag::PStrOffset, pstr_loc) => {
                                                let n0 = cell_as_fixnum!(self.heap[focus+1])
                                                    .get_num() as usize;

                                                if pstr_loc < self.heap.len() - 2 {
                                                    self.heap.pop();
                                                    self.heap.pop();
                                                }

                                                if n == n0 {
                                                    self.pdl.push(pstr_loc_as_cell!(focus));
                                                    self.pdl.push(heap_loc_as_cell!(h));
                                                } else {
                                                    let h_len = self.heap.len();

                                                    self.heap.push(pstr_offset_as_cell!(pstr_loc));
                                                    self.heap.push(fixnum_as_cell!(
                                                        Fixnum::build_with(n as i64)
                                                    ));

                                                    self.pdl.push(pstr_loc_as_cell!(h_len));
                                                    self.pdl.push(heap_loc_as_cell!(h));
                                                }

                                                return;
                                            }
                                            _ => {
                                            }
                                        );

                                        if focus < self.heap.len() - 2 {
                                            self.heap.pop();
                                            self.heap.pop();
                                        }

                                        self.pdl.push(self.heap[focus]);
                                        self.pdl.push(heap_loc_as_cell!(h));

                                        return;
                                    }
                                }

                                break 'outer;
                            }
                            _ => {
                                self.fail = true;
                                break 'outer;
                            }
                        );

                        chars_iter.next();
                    }

                    chars_iter.iter.next();

                    self.pdl.push(chars_iter.iter.focus);
                    self.pdl.push(focus);

                    break;
                }
            }
            PStrCmpResult::Unordered => {
                self.pdl.push(pstr_iter1.focus);
                self.pdl.push(pstr_iter2.focus);
            }
        }

        self.heap.pop();
        self.heap.pop();
    }

    fn unify_list_with_occurs_trigger(
        &mut self,
        l1: usize,
        d2: HeapCellValue,
        mut occurs_trigger: impl FnMut(),
    ) {
        read_heap_cell!(d2,
            (HeapCellValueTag::Lis, l2) => {
                for idx in 0..2 {
                    self.pdl.push(heap_loc_as_cell!(l2+idx));
                    self.pdl.push(heap_loc_as_cell!(l1+idx));
                }
            }
            (HeapCellValueTag::Str, s2) => {
                let (n2, a2) = cell_as_atom_cell!(self.heap[s2])
                    .get_name_and_arity();

                if a2 == 2 && n2 == atom!(".") {
                    for idx in 0..2 {
                        self.pdl.push(heap_loc_as_cell!(s2+1+idx));
                        self.pdl.push(heap_loc_as_cell!(l1+idx));
                    }
                } else {
                    self.fail = true;
                }
            }
            (HeapCellValueTag::PStrLoc | HeapCellValueTag::CStr | HeapCellValueTag::PStr) => {
                self.unify_partial_string_with_occurs_check(
                    list_loc_as_cell!(l1),
                    d2,
                    &mut occurs_trigger,
                )
            }
            (HeapCellValueTag::AttrVar, h) => {
                if self.bind_with_occurs_check(Ref::attr_var(h), list_loc_as_cell!(l1)) {
                    occurs_trigger();
                }
            }
            (HeapCellValueTag::Var, h) => {
                if self.bind_with_occurs_check(Ref::heap_cell(h), list_loc_as_cell!(l1)) {
                    occurs_trigger();
                }
            }
            (HeapCellValueTag::StackVar, s) => {
                if self.bind_with_occurs_check(Ref::stack_cell(s), list_loc_as_cell!(l1)) {
                    occurs_trigger();
                }
            }
            _ => {
                self.fail = true;
            }
        )
    }

    pub(super) fn unify_with_occurs_check_loop(&mut self, mut occurs_trigger: impl FnMut()) {
        let mut tabu_list: IndexSet<(usize, usize)> = IndexSet::new();

        // self.fail = false;

        while !(self.pdl.is_empty() || self.fail) {
            let s1 = self.pdl.pop().unwrap();
            let s1 = self.deref(s1);

            let s2 = self.pdl.pop().unwrap();
            let s2 = self.deref(s2);

            if s1 != s2 {
                let d1 = self.store(s1);
                let d2 = self.store(s2);

                read_heap_cell!(d1,
                    (HeapCellValueTag::AttrVar, h) => {
                        if self.bind_with_occurs_check(Ref::attr_var(h), d2) {
                            occurs_trigger();
                        }
                    }
                    (HeapCellValueTag::Var, h) => {
                        if self.bind_with_occurs_check(Ref::heap_cell(h), d2) {
                            occurs_trigger();
                        }
                    }
                    (HeapCellValueTag::StackVar, s) => {
                        if self.bind_with_occurs_check(Ref::stack_cell(s), d2) {
                            occurs_trigger();
                        }
                    }
                    (HeapCellValueTag::Atom, (name, arity)) => {
                        debug_assert!(arity == 0);
                        self.unify_atom(name, d2);
                    }
                    (HeapCellValueTag::Str, s1) => {
                        if d2.is_constant() {
                            self.fail = true;
                            break;
                        }

                        let s2 = s2.get_value() as usize;

                        if tabu_list.contains(&(s1, s2)) {
                            continue;
                        }

                        self.unify_structure_with_occurs_check(s1, d2, &mut occurs_trigger);

                        if !self.fail {
                            tabu_list.insert((s1, s2));
                        }
                    }
                    (HeapCellValueTag::Lis, l1) => {
                        if d2.is_ref() {
                            let l2 = s2.get_value() as usize;

                            if tabu_list.contains(&(l1, l2)) {
                                continue;
                            }

                            tabu_list.insert((l1, l2));
                        }

                        self.unify_list_with_occurs_trigger(l1, d2, &mut occurs_trigger);
                    }
                    (HeapCellValueTag::PStrLoc, pstr1_loc) => {
                        read_heap_cell!(d2,
                            (HeapCellValueTag::PStrLoc |
                             HeapCellValueTag::Lis |
                             HeapCellValueTag::Str,
                             pstr2_loc) => {
                                if tabu_list.contains(&(pstr1_loc, pstr2_loc)) {
                                    continue;
                                }
                            }
                            (HeapCellValueTag::CStr |
                             HeapCellValueTag::AttrVar |
                             HeapCellValueTag::Var |
                             HeapCellValueTag::StackVar) => {
                            }
                            _ => {
                                self.fail = true;
                                break;
                            }
                        );

                        self.unify_partial_string_with_occurs_check(
                            d1,
                            d2,
                            &mut occurs_trigger,
                        );

                        if !self.fail && !d2.is_constant() {
                            tabu_list.insert((pstr1_loc, d2.get_value()));
                        }
                    }
                    (HeapCellValueTag::CStr) => {
                        read_heap_cell!(d2,
                            (HeapCellValueTag::AttrVar, h) => {
                                self.bind(Ref::attr_var(h), d1);
                                continue;
                            }
                            (HeapCellValueTag::Var, h) => {
                                self.bind(Ref::heap_cell(h), d1);
                                continue;
                            }
                            (HeapCellValueTag::StackVar, s) => {
                                self.bind(Ref::stack_cell(s), d1);
                                continue;
                            }
                            (HeapCellValueTag::Str |
                             HeapCellValueTag::Lis |
                             HeapCellValueTag::PStrLoc) => {
                            }
                            _ => {
                                self.fail = true;
                                return;
                            }
                        );

                        self.unify_partial_string(d2, d1);
                    }
                    (HeapCellValueTag::F64, f1) => {
                        self.unify_f64(f1, d2);
                    }
                    (HeapCellValueTag::Fixnum, n1) => {
                        self.unify_fixnum(n1, d2);
                    }
                    (HeapCellValueTag::Char, c1) => {
                        self.unify_char(c1, d2);
                    }
                    (HeapCellValueTag::Cons, ptr_1) => {
                        self.unify_constant(ptr_1, d2);
                    }
                    _ => {
                        unreachable!();
                    }
                );
            }
        }
    }

    pub(crate) fn read_s(&mut self) -> HeapCellValue {
        match &self.s {
            &HeapPtr::HeapCell(h) => self.deref(self.heap[h]),
            &HeapPtr::PStrChar(h, n) => {
                read_heap_cell!(self.heap[h],
                    (HeapCellValueTag::PStr, pstr_atom) => {
                        let pstr = PartialString::from(pstr_atom);

                        if let Some(c) = pstr.as_str_from(n).chars().next() {
                            char_as_cell!(c)
                        } else { // if has_tail {
                            self.deref(self.heap[h+1]) // heap_loc_as_cell!(h+1)
                        }
                        // } else {
                        //     empty_list_as_cell!()
                        // }
                    }
                    (HeapCellValueTag::CStr, cstr_atom) => {
                        let pstr = PartialString::from(cstr_atom);

                        if let Some(c) = pstr.as_str_from(n).chars().next() {
                            char_as_cell!(c)
                        } else { // if has_tail {
                            empty_list_as_cell!()
                        }
                    }
                    _ => {
                        unreachable!()
                    }
                )
            }
            &HeapPtr::PStrLocation(h, n) => {
                read_heap_cell!(self.heap[h],
                    (HeapCellValueTag::PStr, pstr_atom) => {
                        if n < pstr_atom.len() {
                            let h_len = self.heap.len();

                            self.heap.push(pstr_offset_as_cell!(h));
                            self.heap.push(fixnum_as_cell!(Fixnum::build_with(n as i64)));

                            pstr_loc_as_cell!(h_len)
                        } else {
                            self.deref(self.heap[h+1])
                        }
                    }
                    (HeapCellValueTag::CStr, cstr_atom) => {
                        if n < cstr_atom.len() {
                            let h_len = self.heap.len();

                            self.heap.push(pstr_offset_as_cell!(h));
                            self.heap.push(fixnum_as_cell!(Fixnum::build_with(n as i64)));

                            pstr_loc_as_cell!(h_len)
                        } else {
                            empty_list_as_cell!()
                        }
                    }
                    _ => {
                        unreachable!()
                    }
                )
            }
        }
    }

    pub fn compare_term_test(&mut self) -> Option<Ordering> {
        let mut tabu_list = IndexSet::new();

        while !self.pdl.is_empty() {
            let s1 = self.pdl.pop().unwrap();
            let s1 = self.deref(s1);

            let s2 = self.pdl.pop().unwrap();
            let s2 = self.deref(s2);

            if s1 == s2 {
                continue;
            }

            let v1 = self.store(s1);
            let v2 = self.store(s2);

            let order_cat_v1 = v1.order_category();
            let order_cat_v2 = v2.order_category();

            if order_cat_v1 != order_cat_v2 {
                self.pdl.clear();
                return Some(order_cat_v1.cmp(&order_cat_v2));
            }

            match order_cat_v1 {
                Some(TermOrderCategory::Variable) => {
                    let v1 = v1.as_var().unwrap();
                    let v2 = v2.as_var().unwrap();

                    if v1 != v2 {
                        self.pdl.clear();
                        return Some(v1.cmp(&v2));
                    }
                }
                Some(TermOrderCategory::FloatingPoint) => {
                    let v1 = cell_as_f64_ptr!(v1);
                    let v2 = cell_as_f64_ptr!(v2);

                    if v1 != v2 {
                        self.pdl.clear();
                        return Some(v1.cmp(&v2));
                    }
                }
                Some(TermOrderCategory::Integer) => {
                    let v1 = Number::try_from(v1).unwrap();
                    let v2 = Number::try_from(v2).unwrap();

                    if v1 != v2 {
                        self.pdl.clear();
                        return Some(v1.cmp(&v2));
                    }
                }
                Some(TermOrderCategory::Atom) => {
                    read_heap_cell!(v1,
                        (HeapCellValueTag::Atom, (n1, _a1)) => {
                            read_heap_cell!(v2,
                                (HeapCellValueTag::Atom, (n2, _a2)) => {
                                    if n1 != n2 {
                                        self.pdl.clear();
                                        return Some(n1.cmp(&n2));
                                    }
                                }
                                (HeapCellValueTag::Char, c2) => {
                                    if let Some(c1) = n1.as_char() {
                                        if c1 != c2 {
                                            self.pdl.clear();
                                            return Some(c1.cmp(&c2));
                                        }
                                    } else {
                                        self.pdl.clear();
                                        return Some(Ordering::Greater);
                                    }
                                }
                                _ => {
                                    unreachable!();
                                }
                            )
                        }
                        (HeapCellValueTag::Char, c1) => {
                            read_heap_cell!(v2,
                                (HeapCellValueTag::Atom, (n2, _a2)) => {
                                    if let Some(c2) = n2.as_char() {
                                        if c1 != c2 {
                                            self.pdl.clear();
                                            return Some(c1.cmp(&c2));
                                        }
                                    } else {
                                        self.pdl.clear();
                                        return Some(Ordering::Less);
                                    }
                                }
                                (HeapCellValueTag::Char, c2) => {
                                    if c1 != c2 {
                                        self.pdl.clear();
                                        return Some(c1.cmp(&c2));
                                    }
                                }
                                _ => {
                                    unreachable!()
                                }
                            )
                        }
                        _ => {
                            unreachable!()
                        }
                    )
                }
                Some(TermOrderCategory::Compound) => {
                    fn stalled_pstr_iter_handler(
                        string_iter: HeapPStrIter,
                        stalled_iter: HeapPStrIter,
                        pdl: &mut Vec<HeapCellValue>,
                    ) -> Option<Ordering> {
                        let l = read_heap_cell!(stalled_iter.focus,
                            (HeapCellValueTag::Str, s) => {
                                let (name, arity) = cell_as_atom_cell!(stalled_iter.heap[s])
                                    .get_name_and_arity();

                                if !(name == atom!(".") && arity == 2) {
                                    pdl.clear();
                                    return Some((atom!("."),2).cmp(&(name,arity)));
                                }

                                s+1
                            }
                            (HeapCellValueTag::Lis, l) => {
                                l
                            }
                            _ => {
                                unreachable!()
                            }
                        );

                        let c2 = stalled_iter.heap[l];
                        let c1 = string_iter.chars().next().unwrap();

                        pdl.push(c2);
                        pdl.push(char_as_cell!(c1));

                        None
                    }

                    fn pstr_comparator(
                        heap: &[HeapCellValue],
                        pdl: &mut Vec<HeapCellValue>,
                        s1: usize,
                        s2: usize,
                    ) -> Option<Ordering> {
                        let mut iter1 = HeapPStrIter::new(heap, s1);
                        let mut iter2 = HeapPStrIter::new(heap, s2);

                        match compare_pstr_prefixes(&mut iter1, &mut iter2) {
                            PStrCmpResult::Ordered(ordering) => Some(ordering),
                            _ => {
                                if iter1.num_steps() == 0 && iter2.num_steps() == 0 {
                                    return match iter2.focus.get_tag() {
                                        HeapCellValueTag::CStr | HeapCellValueTag::PStrLoc => {
                                            let result = stalled_pstr_iter_handler(iter2, iter1, pdl);

                                            if let Some(ordering) = result {
                                                Some(ordering.reverse())
                                            } else {
                                                let pdl_len = pdl.len();
                                                pdl.swap(pdl_len - 2, pdl_len - 1);
                                                result
                                            }
                                        }
                                        _ => {
                                            stalled_pstr_iter_handler(iter1, iter2, pdl)
                                        }
                                    };
                                }

                                pdl.push(iter2.focus);
                                pdl.push(iter1.focus);

                                None
                            }
                        }
                    }

                    read_heap_cell!(v1,
                        (HeapCellValueTag::Lis, l1) => {
                            read_heap_cell!(v2,
                                (HeapCellValueTag::CStr | HeapCellValueTag::PStrLoc) => {
                                    let h = self.heap.len();

                                    self.heap.push(v1);
                                    self.heap.push(v2);

                                    if let Some(ordering) = pstr_comparator(
                                        &self.heap, &mut self.pdl, h, h+1
                                    ) {
                                        if ordering != Ordering::Equal {
                                            self.heap.pop();
                                            self.heap.pop();

                                            self.pdl.clear();

                                            return Some(ordering);
                                        }
                                    }

                                    self.heap.pop();
                                    self.heap.pop();
                                }
                                (HeapCellValueTag::Lis, l2) => {
                                    if tabu_list.contains(&(l1, l2)) {
                                        continue;
                                    }

                                    tabu_list.insert((l1, l2));

                                    self.pdl.push(self.heap[l2 + 1]);
                                    self.pdl.push(self.heap[l1 + 1]);

                                    self.pdl.push(self.heap[l2]);
                                    self.pdl.push(self.heap[l1]);
                                }
                                (HeapCellValueTag::Str, s2) => {
                                    if tabu_list.contains(&(l1, s2)) {
                                        continue;
                                    }

                                    let (name, arity) = cell_as_atom_cell!(self.heap[s2])
                                        .get_name_and_arity();

                                    match (atom!("."), 2).cmp(&(name, arity)) {
                                        Ordering::Equal => {
                                            tabu_list.insert((l1, s2));

                                            self.pdl.push(self.heap[s2 + 2]);
                                            self.pdl.push(self.heap[l1 + 1]);

                                            self.pdl.push(self.heap[s2 + 1]);
                                            self.pdl.push(self.heap[l1]);
                                        }
                                        ordering => {
                                            self.pdl.clear();
                                            return Some(ordering);
                                        }
                                    }
                                }
                                _ => {
                                    unreachable!();
                                }
                            )
                        }
                        (HeapCellValueTag::CStr | HeapCellValueTag::PStrLoc) => {
                            let h = self.heap.len();

                            self.heap.push(v1);
                            self.heap.push(v2);

                            if let Some(ordering) = pstr_comparator(
                                &self.heap, &mut self.pdl, h, h+1,
                            ) {
                                if ordering != Ordering::Equal {
                                    self.heap.pop();
                                    self.heap.pop();

                                    self.pdl.clear();

                                    return Some(ordering);
                                }
                            }

                            self.heap.pop();
                            self.heap.pop();
                        }
                        (HeapCellValueTag::Str, s1) => {
                            read_heap_cell!(v2,
                                (HeapCellValueTag::Str, s2) => {
                                    if tabu_list.contains(&(s1, s2)) {
                                        continue;
                                    }

                                    let (n1, a1) = cell_as_atom_cell!(self.heap[s1])
                                        .get_name_and_arity();

                                    let (n2, a2) = cell_as_atom_cell!(self.heap[s2])
                                        .get_name_and_arity();

                                    match (n1,a1).cmp(&(n2,a2)) {
                                        Ordering::Equal => {
                                            tabu_list.insert((s1, s2));

                                            for idx in (1 .. a1+1).rev() {
                                                self.pdl.push(self.heap[s2+idx]);
                                                self.pdl.push(self.heap[s1+idx]);
                                            }
                                        }
                                        ordering => {
                                            self.pdl.clear();
                                            return Some(ordering);
                                        }
                                    }
                                }
                                (HeapCellValueTag::Lis, l2) => {
                                    if tabu_list.contains(&(s1, l2)) {
                                        continue;
                                    }

                                    tabu_list.insert((s1, l2));

                                    let (n1, a1) = cell_as_atom_cell!(self.heap[s1])
                                        .get_name_and_arity();

                                    match (n1,a1).cmp(&(atom!("."), 2)) {
                                        Ordering::Equal => {
                                            self.pdl.push(self.heap[l2]);
                                            self.pdl.push(self.heap[s1+1]);

                                            self.pdl.push(self.heap[l2+1]);
                                            self.pdl.push(self.heap[s1+2]);
                                        }
                                        ordering => {
                                            self.pdl.clear();
                                            return Some(ordering);
                                        }
                                    }
                                }
                                (HeapCellValueTag::CStr | HeapCellValueTag::PStrLoc) => {
                                    let h = self.heap.len();

                                    self.heap.push(v1);
                                    self.heap.push(v2);

                                    if let Some(ordering) = pstr_comparator(
                                        &self.heap, &mut self.pdl, h, h+1,
                                    ) {
                                        if ordering != Ordering::Equal {
                                            self.heap.pop();
                                            self.heap.pop();

                                            self.pdl.clear();

                                            return Some(ordering);
                                        }
                                    }

                                    self.heap.pop();
                                    self.heap.pop();
                                }
                                _ => {
                                    unreachable!()
                                }
                            )
                        }
                        _ => {
                            unreachable!()
                        }
                    );
                }
                None => {
                    if v1 != v2 {
                        self.pdl.clear();
                        return None;
                    }
                }
            }
        }

        Some(Ordering::Equal)
    }

    pub(crate) fn increment_s_ptr(&mut self, rhs: usize) {
        match &mut self.s {
            HeapPtr::HeapCell(ref mut h) => {
                *h += rhs;
            }
            &mut HeapPtr::PStrChar(h, ref mut n) | &mut HeapPtr::PStrLocation(h, ref mut n) => {
                read_heap_cell!(self.heap[h],
                    (HeapCellValueTag::PStr | HeapCellValueTag::CStr, pstr_atom) => {
                        let pstr = PartialString::from(pstr_atom);

                        for c in pstr.as_str_from(*n).chars().take(rhs) {
                            *n += c.len_utf8();
                        }

                        self.s = HeapPtr::PStrLocation(h, *n);
                    }
                    _ => {
                        unreachable!()
                    }
                )
            }
        }
    }

    pub fn match_partial_string(&mut self, value: HeapCellValue, string: Atom, has_tail: bool) {
        let h = self.heap.len();
        self.heap.push(value);

        let mut heap_pstr_iter = HeapPStrIter::new(&self.heap, h);
        let s = string.as_str();

        match heap_pstr_iter.compare_pstr_to_string(s) {
            Some(PStrPrefixCmpResult { focus, offset, prefix_len }) if prefix_len == s.len() => {
                let focus_addr = self.heap[focus];

                read_heap_cell!(focus_addr,
                    (HeapCellValueTag::PStr | HeapCellValueTag::CStr, pstr_atom) => {
                        if has_tail {
                            self.s = HeapPtr::PStrLocation(focus, offset);
                            self.mode = MachineMode::Read;
                        } else if offset == pstr_atom.len() {
                            let focus_addr = heap_pstr_iter.focus;
                            unify!(self, focus_addr, empty_list_as_cell!());
                        } else {
                            self.fail = true;
                        }
                    }
                    (HeapCellValueTag::PStrLoc | HeapCellValueTag::PStrOffset, h) => {
                        if has_tail {
                            let (h, _) = pstr_loc_and_offset(&self.heap, h);

                            self.s = HeapPtr::PStrLocation(h, offset);
                            self.mode = MachineMode::Read;
                        } else {
                            let end_cell = heap_pstr_iter.focus;
                            self.fail = end_cell != empty_list_as_cell!();
                        }
                    }
                    _ => {
                        let focus = heap_pstr_iter.focus();

                        if has_tail {
                            self.s = HeapPtr::HeapCell(focus);
                            self.mode = MachineMode::Read;
                        } else {
                            let focus = heap_pstr_iter.focus;
                            unify!(self, focus, empty_list_as_cell!());
                        }
                    }
                );
            }
            Some(PStrPrefixCmpResult { prefix_len, .. }) => {
                let focus = heap_pstr_iter.focus();
                let tail_addr = self.heap[focus];

                let h = self.heap.len();

                let target_cell = if has_tail {
                    self.s = HeapPtr::HeapCell(h + 1);
                    self.mode = MachineMode::Read;

                    put_partial_string(
                        &mut self.heap,
                        &string.as_str()[prefix_len ..],
                        &mut self.atom_tbl,
                    )
                } else {
                    put_complete_string(
                        &mut self.heap,
                        &string.as_str()[prefix_len ..],
                        &mut self.atom_tbl,
                    )
                };

                unify!(self, tail_addr, target_cell);
            }
            None => {
                self.fail = true;
            }
        }
    }

    pub(super) fn write_literal_to_var(&mut self, deref_v: HeapCellValue, lit: HeapCellValue) {
        let store_v = self.store(deref_v);

        read_heap_cell!(lit,
            (HeapCellValueTag::Atom, (atom, arity)) => {
                if arity == 0 {
                    self.unify_atom(atom, store_v);
                } else {
                    self.fail = true;
                }
            }
            (HeapCellValueTag::Char, c) => {
                self.unify_char(c, store_v);
            }
            (HeapCellValueTag::Fixnum, n) => {
                self.unify_fixnum(n, store_v);
            }
            (HeapCellValueTag::F64, f64_ptr) => {
                self.unify_f64(f64_ptr, store_v);
            }
            (HeapCellValueTag::Cons, ptr) => {
                match_untyped_arena_ptr!(ptr,
                     (ArenaHeaderTag::Integer, n) => {
                         self.unify_big_int(n, store_v);
                     }
                     (ArenaHeaderTag::Rational, r) => {
                         self.unify_rational(r, store_v);
                     }
                     _ => {
                         self.fail = true;
                     }
                )
            }
            (HeapCellValueTag::CStr, cstr_atom) => {
                match store_v.get_tag() {
                    HeapCellValueTag::PStrLoc
                    | HeapCellValueTag::Lis
                    | HeapCellValueTag::Str => {
                        self.match_partial_string(store_v, cstr_atom, false);
                    }
                    _ => {
                        self.fail = true;
                    }
                }
            }
            _ => {
                unreachable!()
            }
        )
    }

    pub(super) fn setup_call_n(&mut self, arity: usize) -> Result<PredicateKey, MachineStub> {
        let addr = self.store(self.deref(self.registers[arity]));

        let (name, narity) = read_heap_cell!(addr,
            (HeapCellValueTag::Str, s) => {
                let (name, narity) = cell_as_atom_cell!(self.heap[s]).get_name_and_arity();

                if narity + arity > MAX_ARITY {
                    let stub = functor_stub(atom!("call"), arity + 1);
                    let err = self.representation_error(RepFlag::MaxArity);
                    return Err(self.error_form(err, stub));
                }

                for i in (1..arity).rev() {
                    self.registers[i + narity] = self.registers[i];
                }

                for i in 1..narity + 1 {
                    self.registers[i] = self.heap[s + i];
                }

                (name, narity)
            }
            (HeapCellValueTag::Atom, (name, arity)) => {
                debug_assert_eq!(arity, 0);
                (name, 0)
            }
            (HeapCellValueTag::Char, c) => {
                (self.atom_tbl.build_with(&c.to_string()), 0)
            }
            (HeapCellValueTag::Var | HeapCellValueTag::AttrVar | HeapCellValueTag::StackVar, _h) => {
                let stub = functor_stub(atom!("call"), arity + 1);
                let err = self.instantiation_error();
                return Err(self.error_form(err, stub));
            }
            _ => {
                let stub = functor_stub(atom!("call"), arity + 1);
                let err = self.type_error(ValidType::Callable, addr);
                return Err(self.error_form(err, stub));
            }
        );

        Ok((name, arity + narity - 1))
    }

    #[inline]
    pub fn is_cyclic_term(&mut self, addr: HeapCellValue) -> bool {
        if addr.is_constant() {
            return false;
        }

        let addr = self.store(self.deref(addr));
        let mut iter = stackful_preorder_iter(&mut self.heap, addr);

        while let Some(value) = iter.next() {
            if value.is_forwarded() {
                let value = heap_bound_store(iter.heap, heap_bound_deref(iter.heap, value));

                if value.is_compound() {
                    return true;
                }
            }
        }

        false
    }

    // arg(+N, +Term, ?Arg)
    pub fn try_arg(&mut self) -> CallResult {
        let stub_gen = || functor_stub(atom!("arg"), 3);
        let n = self.store(self.deref(self.registers[1]));

        read_heap_cell!(n,
            (HeapCellValueTag::Var | HeapCellValueTag::AttrVar | HeapCellValueTag::StackVar) => {
                // 8.5.2.3 a)
                let err = self.instantiation_error();
                return Err(self.error_form(err, stub_gen()));
            }
            _ => {
                let n = match Number::try_from(n) {
                    Ok(Number::Fixnum(n)) => Number::Fixnum(n),
                    Ok(Number::Integer(n)) => Number::Integer(n),
                    _ => {
                        let err = self.type_error(ValidType::Integer, n);
                        return Err(self.error_form(err, stub_gen()));
                    }
                };

                if n < 0 {
                    // 8.5.2.3 e)
                    let err = self.domain_error(DomainErrorType::NotLessThanZero, n);
                    return Err(self.error_form(err, stub_gen()));
                }

                let n = match n {
                    Number::Fixnum(n) => n.get_num() as usize,
                    Number::Integer(n) => n.to_usize().unwrap(),
                    _ => {
                        self.fail = true;
                        return Ok(());
                    }
                };

                let term = self.deref(self.registers[2]);

                read_heap_cell!(self.store(term),
                    (HeapCellValueTag::Var | HeapCellValueTag::AttrVar | HeapCellValueTag::StackVar) => {
                        let err = self.instantiation_error();
                        return Err(self.error_form(err, stub_gen()));
                    }
                    (HeapCellValueTag::Str, o) => {
                        let arity = cell_as_atom_cell!(self.heap[o]).get_arity();

                        if 1 <= n && n <= arity {
                            let a3 = self.registers[3];
                            unify_fn!(*self, a3, heap_loc_as_cell!(o + n));
                        } else {
                            self.fail = true;
                        }
                    }
                    (HeapCellValueTag::Lis, l) => {
                        if n == 1 || n == 2 {
                            let a3 = self.registers[3];
                            unify_fn!(*self, a3, heap_loc_as_cell!(l + n - 1));
                        } else {
                            self.fail = true;
                        }
                    }
                    (HeapCellValueTag::PStrLoc, pstr_loc) => {
                        if n == 1 || n == 2 {
                            let a3 = self.registers[3];
                            let (h, offset) = pstr_loc_and_offset(&self.heap, pstr_loc);

                            let pstr = cell_as_string!(self.heap[h]);
                            let offset = offset.get_num() as usize;

                            if let Some(c) = pstr.as_str_from(offset).chars().next() {
                                if n == 1 {
                                    self.unify_char(c, a3);
                                } else {
                                    let offset = (offset + c.len_utf8()) as i64;
                                    let h_len = self.heap.len();
                                    let pstr_atom: Atom = pstr.into();

                                    if pstr_atom.len() > offset as usize {
                                        self.heap.push(pstr_offset_as_cell!(h));
                                        self.heap.push(fixnum_as_cell!(Fixnum::build_with(offset)));

                                        unify_fn!(*self, pstr_loc_as_cell!(h_len), a3);
                                    } else {
                                        match self.heap[h].get_tag() {
                                            HeapCellValueTag::CStr => {
                                                self.unify_atom(atom!("[]"), self.store(self.deref(a3)));
                                            }
                                            HeapCellValueTag::PStr => {
                                                unify_fn!(*self, self.heap[h+1], a3);
                                            }
                                            _ => {
                                                unreachable!();
                                            }
                                        }
                                    }
                                }
                            } else {
                                unreachable!()
                            }
                        } else {
                            self.fail = true;
                        }
                    }
                    (HeapCellValueTag::CStr, cstr_atom) => {
                        let cstr = PartialString::from(cstr_atom);

                        if let Some(c) = cstr.as_str_from(0).chars().next() {
                            if n == 1 {
                                self.unify_char(c, self.store(self.deref(self.registers[3])));
                            } else if n == 2 {
                                let offset = c.len_utf8() as i64;
                                let h_len = self.heap.len();

                                if cstr_atom.len() > offset as usize {
                                    self.heap.push(atom_as_cstr_cell!(cstr_atom));
                                    self.heap.push(pstr_offset_as_cell!(h_len));
                                    self.heap.push(fixnum_as_cell!(Fixnum::build_with(offset)));

                                    unify_fn!(*self, pstr_loc_as_cell!(h_len+1), self.registers[3]);
                                } else {
                                    self.unify_atom(atom!("[]"), self.store(self.deref(self.registers[3])));
                                }
                            } else {
                                self.fail = true;
                            }
                        } else {
                            unreachable!()
                        }
                    }
                    _ => {
                        // 8.5.2.3 d)
                        let err = self.type_error(ValidType::Compound, term);
                        return Err(self.error_form(err, stub_gen()));
                    }
                )
            }
        );

        Ok(())
    }

    // returns true on failure, false on success.
    pub fn eq_test(&mut self, h1: HeapCellValue, h2: HeapCellValue) -> bool {
        if h1 == h2 {
            return false;
        }

        compare_term_test!(self, h1, h2)
            .map(|o| o != Ordering::Equal)
            .unwrap_or(true)
    }

    pub fn reset_block(&mut self, addr: HeapCellValue) {
        read_heap_cell!(self.store(addr),
            (HeapCellValueTag::Fixnum, n) => {
                self.block = n.get_num() as usize;
            }
            _ => {
                self.fail = true;
            }
        )
    }

    #[inline(always)]
    fn try_functor_compound_case(&mut self, name: Atom, arity: usize) {
        self.try_functor_unify_components(atom_as_cell!(name), arity);
    }

    fn try_functor_unify_components(&mut self, name: HeapCellValue, arity: usize) {
        let a2 = self.deref(self.registers[2]);
        self.write_literal_to_var(a2, name);

        if !self.fail {
            let a3 = self.store(self.deref(self.registers[3]));
            self.unify_fixnum(Fixnum::build_with(arity as i64), a3);
        }
    }

    fn try_functor_fabricate_struct(&mut self, name: Atom, arity: usize, r: Ref) {
        let h = self.heap.len();

        let f_a = if name == atom!(".") && arity == 2 {
            self.heap.push(heap_loc_as_cell!(h));
            self.heap.push(heap_loc_as_cell!(h+1));

            list_loc_as_cell!(h)
        } else {
            self.heap.push(atom_as_cell!(name, arity));

            for i in 0..arity {
                self.heap.push(heap_loc_as_cell!(h + i + 1));
            }

            if arity == 0 {
                heap_loc_as_cell!(h)
            } else {
                str_loc_as_cell!(h)
            }
        };

        (self.bind_fn)(self, r, f_a);
    }

    pub fn try_functor(&mut self) -> CallResult {
        let stub_gen = || functor_stub(atom!("functor"), 3);
        let a1 = self.store(self.deref(self.registers[1]));

        read_heap_cell!(a1,
            (HeapCellValueTag::Cons | HeapCellValueTag::Char | HeapCellValueTag::Fixnum |
             HeapCellValueTag::F64) => {
                self.try_functor_unify_components(a1, 0);
            }
            (HeapCellValueTag::Atom, (_name, arity)) => {
                debug_assert_eq!(arity, 0);
                self.try_functor_unify_components(a1, 0);
            }
            (HeapCellValueTag::Str, s) => {
                let (name, arity) = cell_as_atom_cell!(self.heap[s]).get_name_and_arity();
                self.try_functor_compound_case(name, arity);
            }
            (HeapCellValueTag::Lis | HeapCellValueTag::PStrLoc | HeapCellValueTag::CStr) => {
                self.try_functor_compound_case(atom!("."), 2);
            }
            (HeapCellValueTag::Var | HeapCellValueTag::AttrVar | HeapCellValueTag::StackVar) => {
                let deref_name = self.deref(self.registers[2]);
                let store_name = self.store(deref_name);

                let arity = self.store(self.deref(self.registers[3]));

                if store_name.is_var() || arity.is_var() {
                    // 8.5.1.3 a) & 8.5.1.3 b)
                    let err = self.instantiation_error();
                    return Err(self.error_form(err, stub_gen()));
                }

                let arity = match Number::try_from(arity) {
                    Ok(Number::Fixnum(n)) => Some(n.get_num()),
                    Ok(Number::Integer(n)) => n.to_i64(),
                    Ok(Number::Rational(n)) if n.denom() == &1 => n.numer().to_i64(),
                    _ => {
                        let err = self.type_error(ValidType::Integer, arity);
                        return Err(self.error_form(err, stub_gen()));
                    }
                };

                let arity = match arity {
                    Some(arity) => arity,
                    None => {
                        self.fail = true;
                        return Ok(());
                    }
                };

                if arity > MAX_ARITY as i64 {
                    // 8.5.1.3 f)
                    let err = self.representation_error(RepFlag::MaxArity);
                    return Err(self.error_form(err, stub_gen()));
                } else if arity < 0 {
                    // 8.5.1.3 g)
                    let arity = Number::Fixnum(Fixnum::build_with(arity));
                    let err = self.domain_error(DomainErrorType::NotLessThanZero, arity);

                    return Err(self.error_form(err, stub_gen()));
                }

                read_heap_cell!(store_name,
                    (HeapCellValueTag::Cons | HeapCellValueTag::Char | HeapCellValueTag::Fixnum |
                     HeapCellValueTag::F64) if arity == 0 => {
                        self.bind(a1.as_var().unwrap(), deref_name);
                    }
                    (HeapCellValueTag::Atom, (name, atom_arity)) => {
                        debug_assert_eq!(atom_arity, 0);
                        self.try_functor_fabricate_struct(
                            name,
                            arity as usize,
                            a1.as_var().unwrap(),
                        );
                    }
                    (HeapCellValueTag::Char, c) => {
                        let c = self.atom_tbl.build_with(&c.to_string());

                        self.try_functor_fabricate_struct(
                            c,
                            arity as usize,
                            a1.as_var().unwrap(),
                        );
                    }
                    _ => {
                        let err = self.type_error(ValidType::Atomic, store_name);
                        return Err(self.error_form(err, stub_gen()));
                    } // 8.5.1.3 c)
                );
            }
            _ => {
                self.fail = true;
            }
        );

        Ok(())
    }

    pub fn try_from_list(
        &mut self,
        value: HeapCellValue,
        stub_gen: impl Fn() -> FunctorStub,
    ) -> Result<Vec<HeapCellValue>, MachineStub> {
        let deref_v = self.deref(value);
        let store_v = self.store(deref_v);

        read_heap_cell!(store_v,
            (HeapCellValueTag::Lis, l) => {
                self.try_from_inner_list(vec![], l, stub_gen, store_v)
            }
            (HeapCellValueTag::PStrLoc, h) => {
                self.try_from_partial_string(vec![], h, stub_gen, store_v)
            }
            (HeapCellValueTag::AttrVar | HeapCellValueTag::StackVar | HeapCellValueTag::Var) => {
                let err = self.instantiation_error();
                Err(self.error_form(err, stub_gen()))
            }
            (HeapCellValueTag::Atom, (name, arity)) => {
                if name == atom!("[]") && arity == 0 {
                    Ok(vec![])
                } else {
                    let err = self.type_error(ValidType::List, store_v);
                    Err(self.error_form(err, stub_gen()))
                }
            }
            (HeapCellValueTag::CStr, cstr_atom) => {
                let cstr = cstr_atom.as_str();
                Ok(cstr.chars().map(|c| char_as_cell!(c)).collect())
            }
            _ => {
                let err = self.type_error(ValidType::List, store_v);
                Err(self.error_form(err, stub_gen()))
            }
        )
    }

    fn try_from_inner_list(
        &mut self,
        mut result: Vec<HeapCellValue>,
        mut l: usize,
        stub_gen: impl Fn() -> FunctorStub,
        a1: HeapCellValue,
    ) -> Result<Vec<HeapCellValue>, MachineStub> {
        result.push(self.heap[l]);
        l += 1;

        loop {
            let deref_v = self.deref(self.heap[l]);
            let store_v = self.store(self.heap[l]);

            read_heap_cell!(store_v,
                (HeapCellValueTag::Lis, hcp) => {
                    result.push(self.heap[hcp]);
                    l = hcp + 1;
                }
                (HeapCellValueTag::PStrOffset) => {
                    return self.try_from_partial_string(result, deref_v.get_value(), stub_gen, a1);
                }
                (HeapCellValueTag::Atom, (name, arity)) => {
                    if name == atom!("[]") && arity == 0 {
                        break;
                    } else {
                        let err = self.type_error(ValidType::List, a1);
                        return Err(self.error_form(err, stub_gen()));
                    }
                }
                _ => {
                    if store_v.is_var() {
                        let err = self.instantiation_error();
                        return Err(self.error_form(err, stub_gen()));
                    } else {
                        let err = self.type_error(ValidType::List, a1);
                        return Err(self.error_form(err, stub_gen()));
                    }
                }
            );
        }

        Ok(result)
    }

    fn try_from_partial_string(
        &mut self,
        mut chars: Vec<HeapCellValue>,
        h: usize,
        stub_gen: impl Fn() -> FunctorStub,
        a1: HeapCellValue,
    ) -> Result<Vec<HeapCellValue>, MachineStub> {
        let mut heap_pstr_iter = HeapPStrIter::new(&self.heap, h);

        while let Some(iteratee) = heap_pstr_iter.next() {
            match iteratee {
                PStrIteratee::Char(_, c) =>
                    chars.push(char_as_cell!(c)),
                PStrIteratee::PStrSegment(_, pstr_atom, n) => {
                    let pstr = PartialString::from(pstr_atom);
                    chars.extend(pstr.as_str_from(n).chars().map(|c| char_as_cell!(c)));
                }
            }
        }

        match self.heap[h].get_tag() {
            HeapCellValueTag::PStr => {
                if heap_pstr_iter.at_string_terminator() {
                    Ok(chars)
                } else {
                    read_heap_cell!(self.heap[heap_pstr_iter.focus()],
                        (HeapCellValueTag::Lis, l) => {
                            self.try_from_inner_list(chars, l, stub_gen, a1)
                        }
                        (HeapCellValueTag::Atom, (name, arity)) => {
                            if name == atom!(".") && arity == 2 {
                                let l = heap_pstr_iter.focus() + 1;
                                self.try_from_inner_list(chars, l, stub_gen, a1)
                            } else {
                                let err = self.type_error(ValidType::List, a1);
                                Err(self.error_form(err, stub_gen()))
                            }
                        }
                        _ => {
                            let err = self.type_error(ValidType::List, a1);
                            Err(self.error_form(err, stub_gen()))
                        }
                    )
                }
            }
            HeapCellValueTag::CStr => Ok(chars),
            _ => {
                unreachable!()
            }
        }
    }

    // returns true on failure.
    pub fn ground_test(&mut self) -> bool {
        if self.registers[1].is_constant() {
            return false;
        }

        let value = self.store(self.deref(self.registers[1]));

        if value.is_stack_var() {
            return true;
        }

        for v in stackful_preorder_iter(&mut self.heap, value) {
            let v = unmark_cell_bits!(v);

            if v.is_var() {
                return true;
            }
        }

        false
    }

    pub fn integers_to_bytevec(
        &mut self,
        value: HeapCellValue,
        stub_gen: impl Fn() -> FunctorStub,
    ) -> Vec<u8> {
        let mut bytes: Vec<u8> = Vec::new();

        match self.try_from_list(value, stub_gen) {
            Err(_) => {
                unreachable!()
            }
            Ok(addrs) => {
                for addr in addrs {
                    let addr = self.store(self.deref(addr));

                    match Number::try_from(addr) {
                        Ok(Number::Fixnum(n)) => match u8::try_from(n.get_num()) {
                            Ok(b) => bytes.push(b),
                            Err(_) => {}
                        },
                        Ok(Number::Integer(n)) => {
                            if let Some(b) = n.to_u8() {
                                bytes.push(b);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        bytes
    }

    // see 8.4.4.3 of Draft Technical Corrigendum 2 for an error guide.
    pub fn project_onto_key(&mut self, value: HeapCellValue) -> Result<HeapCellValue, MachineStub> {
        let stub_gen = || functor_stub(atom!("keysort"), 2);
        let store_v = self.store(self.deref(value));

        if store_v.is_var() {
            let err = self.instantiation_error();
            return Err(self.error_form(err, stub_gen()));
        }

        read_heap_cell!(store_v,
            (HeapCellValueTag::Str, s) => {
                let (name, arity) = cell_as_atom_cell!(self.heap[s]).get_name_and_arity();

                if name == atom!("-") && arity == 2 {
                    Ok(heap_loc_as_cell!(s + 1))
                } else {
                    let err = self.type_error(ValidType::Pair, self.heap[s]);
                    Err(self.error_form(err, stub_gen()))
                }
            }
            _ => {
                let err = self.type_error(ValidType::Pair, store_v);
                Err(self.error_form(err, stub_gen()))
            }
        )
    }

    /*
    pub fn setup_built_in_call(&mut self, ct: BuiltInClauseType) {
        self.num_of_args = ct.arity();
        self.b0 = self.b;

        self.p = CodePtr::BuiltInClause(ct, self.p.local());
    }
    */

    pub fn deallocate(&mut self) {
        let e = self.e;
        let frame = self.stack.index_and_frame(e);

        self.cp = frame.prelude.cp;
        self.e = frame.prelude.e;

        if e > self.b {
            self.stack.truncate(e);
        }

        self.p += 1;
    }

    pub fn throw_interrupt_exception(&mut self) {
        let err = self.interrupt_error();
        let src = functor_stub(atom!("repl"), 0);
        let err = self.error_form(err, src);

        self.throw_exception(err);
    }
}
