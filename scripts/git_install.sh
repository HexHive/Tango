#!/usr/bin/env bash

# defaults
batch=false
pythonminversion="3.7"
pipminversion="22.2"
clangminversion="13"
pipflags="-e" # install package in edit mode, for development
setup_venv=true
verify_deps=true

# this is needed to prevent packages like tzdata from blocking installation
export DEBIAN_FRONTEND=noninteractive

while getopts yDVP:p:C:F: opt
do
    case "${opt}" in
        y) batch=true;;
        D) setup_venv=false;;
        V) verify_deps=false;;
        P) pythonminversion="${OPTARG}";;
        p) pipminversion="${OPTARG}";;
        C) clangminversion="${OPTARG}";;
        F) pipflags="${OPTARG}";;
    esac
done
shift $((OPTIND-1))

prompt() {
    if $batch; then
        >&2 echo "$1 Y"
        >&2 echo "Proceeding in batch mode..."
        return 0
    else
        >&2 read -p "$1 (Y/n): " confirm && \
            [[ $confirm == [nN] || $confirm == [nN][oO] ]]
        [ $? -eq 1 ]
        return $?
    fi
}

find_binary() {
    for name in "$@"; do
        if bin="$(which $name)"; then
            echo "$bin"
            return 0
        fi
    done
    return 1
}

find_python() {
    if [ "$PYTHONBIN" == "" ]; then
        if [ "$PYTHONINSTALLED" == "" ]; then
            if ! pybin="$(find_binary python$pythonminversion python3 python)";
            then
                return 1
            fi
        else
            pybin="$PYTHONINSTALLED"
        fi
    else
        pybin="$PYTHONBIN"
    fi
    echo "$pybin"
    return 0
}

find_pip() {
    if [ "$PIPBIN" == "" ]; then
        if [ "$PIPINSTALLED" == "" ]; then
            if ! pybin="$(find_python)"; then
                if ! pipbin="$(find_binary pip3 pip)"; then
                    return 1
                fi
            else
                pipbin="$pybin -m pip"
            fi
        else
            pipbin="$PIPINSTALLED"
        fi
    else
        pipbin="$PIPBIN"
    fi
    echo "$pipbin"
    return 0
}

# from SO: https://stackoverflow.com/a/4025065
vercomp () {
    if [[ $1 == $2 ]]
    then
        return 0
    fi
    local IFS=.
    local i ver1=($1) ver2=($2)
    # fill empty fields in ver1 with zeros
    for ((i=${#ver1[@]}; i<${#ver2[@]}; i++))
    do
        ver1[i]=0
    done
    for ((i=0; i<${#ver1[@]}; i++))
    do
        if [[ -z ${ver2[i]} ]]
        then
            # fill empty fields in ver2 with zeros
            ver2[i]=0
        fi
        if ((10#${ver1[i]} > 10#${ver2[i]}))
        then
            return 1
        fi
        if ((10#${ver1[i]} < 10#${ver2[i]}))
        then
            return 2
        fi
    done
    return 0
}

test_version () {
    op=$2
    pass=1
    vercomp $1 $3
    case $? in
        0 ) [[ $op =~ [\<\>\=]'=' ]] && pass=0;;
        1 ) ([[ $op =~ '>'[\=]? ]] || [[ $op == '!=' ]]) && pass=0;;
        2 ) ([[ $op =~ '<'[\=]? ]] || [[ $op == '!=' ]]) && pass=0;;
    esac
    return $pass
}

test_version_spec() {
    version=$1
    IFS=,
    specs=($2)
    unset IFS

    for spec in "${specs[@]}"; do
        specop=$(echo $spec | grep -oP '[<>=!]+(?=.*)')
        specver=$(echo $spec | grep -oP '[<>=!]+\K.*')
        if ! test_version $version $specop $specver; then
            return 1
        fi
    done
    return 0
}

# from SO: https://unix.stackexchange.com/a/35265
find_up() {
    path="$1"
    shift 1
    while [[ "$path" != / ]];
    do
        result="$(find "$path" -maxdepth 1 -mindepth 1 "$@")"
        if [ "$result" != "" ]; then
            echo "$result"
            return 0
        fi
        # Note: if you want to ignore symlinks, use "$(realpath -s "$path"/..)"
        path="$(readlink -f "$path"/..)"
    done
    return 1
}

find_pkgroot() {
    if [ "$1" == "" ]; then
        path=.
    else
        path="$1"
    fi
    if ! toml="$(find_up "$path" -name pyproject.toml)"; then
        return 1
    fi
    pkgroot="$(dirname "$toml")"
    echo "$pkgroot"
    return 0
}

check_python_version() {
    if [ "$1" != "" ]; then
        python="$1"
    elif ! python="$(find_python)"; then
        >&2 echo "Failed to find python binary."
        return 1
    fi
    pyver="$($python --version | cut -d' ' -f2 -s)"
    if ! test_version "$pyver" '>=' "$pythonminversion"; then
        >&2 echo "Python version $pyver is not in '>=$pythonminversion'."
        return 1
    fi
    pipver="$($python -c \
        'import pkg_resources as pkg; print(pkg.get_distribution("pip").version)' \
        2>/dev/null)"
    if ! [ $? -eq 0 ]; then
        >&2 echo "pip is not installed; consider installing python3-pip."
        return 2
    fi
    if ! test_version "$pipver" '>=' "$pipminversion"; then
        >&2 echo "pip version $pipver is not in >=$pipminversion."
        return 2
    fi
    pip="$(find_pip)"

    if $verify_deps; then
        if ! pkgroot="$(find_pkgroot)"; then
            >&2 echo "Failed to find package root. Make sure the current working" \
                "directory, or one of its ancestors, contains pyproject.toml."
            return 99
        fi
        # first check if there's a problem with resolving dependencies
        log="$($pip --python $python install --dry-run --ignore-requires-python \
            $pipflags "$pkgroot" 2>&1)"
        if ! [ $? -eq 0 ]; then
            >&2 echo "Failed to resolve package dependencies. Check pip output:"
            >&2 echo "$log"
            return 99
        fi
        # then check if problems arise from python version
        log="$($pip --python $python install --dry-run $pipflags "$pkgroot" \
            2>&1 1>/dev/null)"
        if ! [ $? -eq 0 ]; then
            >&2 echo "Python version $pyver is too low. Check pip output:"
            >&2 echo "$log"
            requires_python="$(echo "$log" | \
                grep -oP '(\d+[.]?)+? not in '"'"'\K(.*?)(?='"'"')')"
            echo "$requires_python"
            return 3
        fi
    fi
    echo "$python"
    return 0
}

check_clang_version() {
    if ! clang="$(find_binary clang-$clangminversion clang)"; then
        >&2 echo "Failed to find clang binary."
        return 1
    fi
    clangver="$($clang --version | \
        grep -o -E "[[:digit:]]+.[[:digit:]]+.[[:digit:]]+" | head -n 1)"
    if ! test_version "$clangver" '>=' "$clangminversion"; then
        >&2 echo "clang version $clangver is not in >=$clangminversion."
        return 1
    fi
    return 0
}

install_python() {
    requires_python="$1"

    (set -x; sudo add-apt-repository -yu ppa:deadsnakes/ppa >&/dev/null)
    ver=$(apt-cache search -n '^python(\d+[.]?)+?' | \
            grep -oP '^python\K(\d+[.]?)+(?= - )' | uniq | sort -Vr | (
        while read -r ver;
        do
            if test_version_spec $ver $requires_python; then
                echo "$ver"
                exit 0
            fi
        done && exit 1)
    )
    if ! [ $? -eq 0 ]; then
        return 1
    fi

    >&2 echo "Python $ver found! Installing..."
    (set -x; sudo -E apt-get install -y python$ver python$ver-venv \
        python$ver-dev python3-pip >&/dev/null)
    python="$(which python$ver)"
    echo "$python"
}

install_clang() {
    v=$clangminversion
    (
        set -xe;
        LLVM_VERSION=13
        sudo apt-get install -y curl wget
        curl -sL https://apt.llvm.org/llvm.sh | sudo bash -s $LLVM_VERSION
        sudo apt-get install -y libc++-$v-dev libc++abi-$v-dev
    )
    return $?
}

install_make() {
    (
        set -xe
        sudo apt-get update
        sudo -E apt-get install -y make
    )
    return $?
}

install_cmake() {
    (
        set -xe
        sudo apt-get update
        sudo -E apt-get install -y cmake
    )
    return $?
}

install_graphviz() {
    (
        set -xe
        sudo apt-get update
        sudo -E apt-get install -y graphviz
    )
    return $?
}

while : ; do
    requires_python="$(check_python_version)"
    case $? in
        0 ) python="$requires_python"
            >&2 echo "Found suitable python binary at $python"
            break
            ;;
        1 ) >&2 echo "Could not find suitable python binary!"
            if ! prompt "Attempt to install python >=$pythonminversion?"; then
                echo "Aborting installation!"
                exit 1
            fi
            if ! PYTHONINSTALLED="$(install_python ">=$pythonminversion")";
            then
                >&2 echo "Could not find suitable python version." \
                    "Aborting installation!"
                exit 1
            fi
            export PYTHONINSTALLED
            continue
            ;;
        2 ) if ! prompt "Attempt to find and install python3-pip?"; then \
                echo "Aborting installation!"
                exit 1
            fi
            (set -x; sudo apt-get update)
            if ! (set -x; sudo -E apt-get install -y python3-pip); then
                >&2 echo "Failed to install pip. Aborting installation!"
                exit 1
            fi
            pip="$(find_pip)"
            if ! (set -x; $pip install --upgrade pip); then
                >&2 echo "Failed to upgrade pip. Aborting installation!"
                exit 1
            fi
            continue
            ;;
        3 ) >&2 echo "Installed python version does not match requirement" \
                "'$requires_python'"
            if ! prompt "Attempt to find and install suitable version?"; then
                echo "Aborting installation!"
                exit 1
            fi
            >&2 echo "Looking for python version: $requires_python"

            if ! PYTHONINSTALLED="$(install_python "$requires_python")";
            then
                >&2 echo "Could not find suitable python version." \
                    "Aborting installation!"
                exit 1
            fi
            export PYTHONINSTALLED

            >&2 echo "Installed suitable python binary at $PYTHONINSTALLED"
            continue
            ;;
        99) >&2 echo "Fatal error. Aboting installation!"
            exit 1
            ;;
    esac
done

# from this point forward, $python holds the path to the correct python binary
if ! check_clang_version; then
    >&2 echo "Clang $clangminversion is required for coverage instrumentation."
    if prompt "Proceed?" && ! install_clang; then
        >&2 echo "Failed to install clang. Aborting installation!"
        exit 1
    fi
fi

if ! find_binary make; then
    >&2 echo "Make is needed for building targets."
    if prompt "Proceed?" && ! install_make; then
        >&2 echo "Failed to install make. Aborting installation!"
        exit 1
    fi
fi

if ! find_binary cmake; then
    >&2 echo "CMake is needed for compiling some targets."
    if prompt "Proceed?" && ! install_cmake; then
        >&2 echo "Failed to install cmake. Aborting installation!"
        exit 1
    fi
fi

if ! find_binary dot; then
    >&2 echo "Graphviz is needed for displaying the WebUI."
    if prompt "Proceed?" && ! install_graphviz; then
        >&2 echo "Failed to install graphviz. Aborting installation!"
        exit 1
    fi
fi

if ! $setup_venv; then
    exit 0;
fi

pip="$(find_pip)"
set -e

last_venv=""
while : ;
do
    if [ "$VIRTUAL_ENV" == "" ]; then
        if ! venv="$(find_up . -name .venv -type d)" || \
                [ "$venv" == "$last_venv" ]; then
            pkgroot="$(find_pkgroot)"
            venv="$pkgroot"/.venv
            >&2 echo "Creating virtual environment at $venv"
            $python -m venv "$venv"
            source "$venv"/bin/activate
            break
        else
            >&2 echo "Checking existing .venv at $venv"
            source "$venv"/bin/activate
            continue
        fi
    else
        vpython="$(find_binary python)"
        if ! check_python_version "$vpython"; then
            last_venv="$VIRTUAL_ENV"
            >&2 echo "Activated virtual environment is incompatible." \
                "We'll resort to creating a new one."
            deactivate
            continue
        else
            >&2 echo "Using activated virtual environment at $VIRTUAL_ENV"
            venv="$VIRTUAL_ENV"
            break
        fi
    fi
done

>&2 echo "Installing package in virtual environment $venv"
pip install --upgrade pip
pkgroot="$(find_pkgroot)"
if ! (pip install $pipflags "$pkgroot" && \
        pip install "$pkgroot"[complete]); then
    >&2 echo "Failed to install package in virtual environment. Aborting!"
    exit 1
fi

>&2 echo "Building native C libraries"
make -C "$pkgroot"/lib all

if [ "$TANGO_LIBDIR" == "" ]; then
    echo "export TANGO_LIBDIR='$(realpath "$pkgroot/lib")'" >> "$venv"/bin/activate
fi

>&2 echo "Installation complete!"
