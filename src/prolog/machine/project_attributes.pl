driver(QueryVars, AttrVars) :-
    gather_modules(AttrVars, Modules0, _),
    sort(Modules0, Modules),
    call_project_attributes(Modules, QueryVars, AttrVars),
    call_attribute_goals(Modules, QueryVars),
    call_attribute_goals(Modules, AttrVars),
    '$deallocate'.

call_project_attributes([], _, _).
call_project_attributes([Module|Modules], QueryVars, AttrVars) :-
    (   catch(Module:project_attributes(QueryVars, AttrVars),
	      error(evaluation_error((Module:project_attributes)/2), project_attributes/2),
	      true) -> true
    ;   true
    ),
    call_project_attributes(Modules, QueryVars, AttrVars).

call_attribute_goals([], _).
call_attribute_goals([Module | Modules], AttrVars) :-
    call_goals(AttrVars, Module),
    call_attribute_goals(Modules, AttrVars).

call_goals([], _).
call_goals([AttrVar|AttrVars], Module) :-
    (   catch(Module:attribute_goals(AttrVar, Goal),
	      error(evaluation_error((Module:attribute_goals)/2), attribute_goals/2),
	      true),
	nonvar(Goal) -> nl, writeq(Goal)
    ;   true
    ),
    call_goals(AttrVars, Module).

gather_modules([], [], _).
gather_modules([AttrVar|AttrVars], Modules, Modules0) :-
    '$get_attr_list'(AttrVar, Attrs),
    gather_modules_for_attrs(Attrs, Modules, Modules0),
    gather_modules(AttrVars, Modules0, _).

gather_modules_for_attrs(Attrs, Modules, Modules) :-
    var(Attrs), !.
gather_modules_for_attrs([Attr|Attrs], [Module|Modules], Modules0) :-
    '$module_of'(Module, Attr),
    gather_modules_for_attrs(Attrs, Modules, Modules0).
