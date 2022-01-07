use static_string_indexing::index_static_strings;
use instructions_template::generate_instructions_rs;

use std::env;
use std::fs;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::process::Command;

fn find_prolog_files(libraries: &mut File, prefix: &str, current_dir: &Path) {
    let entries = match current_dir.read_dir() {
        Ok(entries) => entries,
        Err(_) => return,
    };

    for entry in entries.filter_map(Result::ok).map(|e| e.path()) {
        if entry.is_dir() {
            if let Some(file_name) = entry.file_name() {
                let new_prefix = prefix.to_owned() + file_name.to_str().unwrap() + "/";
                find_prolog_files(libraries, &new_prefix, &entry);
            }
        } else if entry.is_file() {
            let ext = std::ffi::OsStr::new("pl");
            if entry.extension() == Some(ext) {
                let contain = String::from_utf8(fs::read(&entry).unwrap()).unwrap();
                let name = entry.file_stem().unwrap().to_str().unwrap();

                let line = format!(
                    "        m.insert(\"{}\",\n{:?});\n",
                    prefix.to_owned() + name,
                    contain
                );

                libraries.write_all(line.as_bytes()).unwrap();
            }
        }
    }
}

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("libraries.rs");

    let mut libraries = File::create(&dest_path).unwrap();
    let lib_path = Path::new("src/lib");

    libraries
        .write_all(
            b"ref_thread_local::ref_thread_local! {
    pub(crate) static managed LIBRARIES: IndexMap<&'static str, &'static str> = {
        let mut m = IndexMap::new();\n",
        )
        .unwrap();

    find_prolog_files(&mut libraries, "", &lib_path);
    libraries.write_all(b"\n        m\n    };\n}\n").unwrap();

    let instructions_path = Path::new("src/instructions.rs");
    let mut instructions_file = File::create(&instructions_path).unwrap();

    let quoted_output = generate_instructions_rs();

    instructions_file
        .write_all(quoted_output.to_string().as_bytes())
        .unwrap();

    Command::new("rustfmt")
        .arg(instructions_path.as_os_str())
        .spawn().unwrap()
        .wait().unwrap();

    let static_atoms_path = Path::new("src/static_atoms.rs");
    let mut static_atoms_file = File::create(&static_atoms_path).unwrap();

    let quoted_output = index_static_strings();

    static_atoms_file
        .write_all(quoted_output.to_string().as_bytes())
        .unwrap();

    Command::new("rustfmt")
        .arg(static_atoms_path.as_os_str())
        .spawn().unwrap()
        .wait().unwrap();
}
