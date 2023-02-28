# Let's Tango!

## Usage

Clone the git repo to get started.

```
git clone git@github.com:HexHive/tangofuzz.git tango && cd tango
git submodule init
git submodule update scripts/ # we don't need the other submodules for now
```

### Development environment (Ubuntu 18.04+ only)

First, install dependencies locally at the risk of slightly littering your
package manager (the script below takes care of finding and installing the
correct python version and setting up a virtualenv).

```
./scripts/git_install.sh
```

After installation, go back to the repo root.

```
pushd targets; USE_ASAN=1 make expat/; popd
```

Then, activate the venv and grant the capabilities to Python once.

```
source .venv/bin/activate
sudo setcap cap_net_admin,cap_sys_admin,cap_dac_override+eip $(realpath "$(which python)")
```

Finally, launch a campaign and watch the graph at http://localhost:8080.

```
python main.py -v targets/expat/fuzz.json
```

### Docker environment (easier, but more hands-off)

Build the docker images.

```
pushd scripts && sudo docker build -t tango:latest . && popd
```

Then, from the repo root, start the container and watch the graph at
http://localhost:8080.

```
scripts/docker_run.sh expat
```
