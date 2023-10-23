from importlib.util import find_spec
from os.path import exists
from subprocess import run

is_portable = True if exists("python_embeded") else False

def check_and_install_package(package_name: str, install_name: str = None) -> None:
    if find_spec(package_name):
        return

    print(f"/_\ Installing {package_name}")

    package = install_name if install_name else package_name

    if is_portable:
        command = f".\\python_embeded\\python.exe -s -m pip install {package}"
    else:
        command = f"pip install {package}"

    process = run(command, shell=True, check=True, capture_output=True)

print("/_\ Checking packages")

check_and_install_package("transformers")
check_and_install_package("pillow")
check_and_install_package("matplotlib")
check_and_install_package("numpy")
check_and_install_package("scipy")
check_and_install_package("fastapi")
check_and_install_package("pytorch_lightning")
check_and_install_package("clip", "git+https://github.com/openai/CLIP.git")