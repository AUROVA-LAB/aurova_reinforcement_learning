./isaaclab.sh -p -m pip install -e source/isaaclab
./isaaclab.sh -p -m pip install --no-build-isolation -e source/isaaclab
./isaaclab.sh -p -m pip install --upgrade --force-reinstall setuptools wheel pip
./isaaclab.sh -p -m pip install --no-build-isolation -e source/isaaclab
./isaaclab.sh -p -m pip install --upgrade --force-reinstall setuptools wheel pip
./isaaclab.sh -p -m pip install "packaging==23.2"
./isaaclab.sh -p -m pip check
pip install --upgrade pip setuptools wheel
./isaaclab.sh -p -m pip install pip==24.0 setuptools==68.0 wheel==0.42 packaging==23.2
./isaaclab.sh -p -m pip install --no-build-isolation -e source/isaaclab
./isaaclab.sh -p -m pip install "pip==24.0" "setuptools==68.0.0" "wheel==0.42.0"
./isaaclab.sh -p -m pip install --no-cache-dir evdev==1.6.1
./isaaclab.sh -p -m pip install --no-build-isolation -e source/isaaclab
./isaaclab.sh -p -m pip install --force-reinstall "typing_extensions==4.12.2"
rm -rf /isaac-sim/extscache/omni.kit.pip_archive*
./isaaclab.sh -p -c "from typing_extensions import deprecated"
pip install toml


