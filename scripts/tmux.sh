#!/usr/bin/env bash
set -e

# defaults
count=4
duration= # no limit
sessname='tango'
label= # no label
overrides=()
outdir='./ar'
venv='./.venv'
cwd="$(pwd)"
detach=false
htop=false
interactive=false
notify=false

while getopts c:d:s:l:o:O:v:C:DTin opt
do
	case "${opt}" in
		c) count=${OPTARG};;
		d) duration=${OPTARG};;
		s) sessname=${OPTARG};;
		l) label=_${OPTARG};;
		o) overrides+=("${OPTARG}");;
		O) outdir="${OPTARG}";;
		v) venv="${OPTARG}";;
		C) cwd="${OPTARG}";;
		D) detach=true;;
		T) htop=true;;
		i) interactive=true;;
		n) notify=true;;
	esac
done

args=()
if [ "$OPTIND" -ge 2 ]; then
		shift "$((OPTIND - 2))"
		last="$1"
		shift 1
else
		shift "$((OPTIND - 1))"
fi

if [ "$last" = "--" ]; then
	args=("$@")
else
	targets=("$@")
fi

tmux start-server

if ! tmux has-session -t "$sessname" >&/dev/null; then
	tmux new-session -d -s "$sessname"
	tmux set default-terminal 'tmux-256color'
	if $htop; then
		tmux new-window -k -t "$sessname:0" -n 'htop' \
			'/usr/bin/env htop'
		windex=1
	else
		windex=0
	fi
	new=true
else
	windex=$(tmux list-windows -t $sessname -F '#{window_index}' | sort -n \
		| tail -n1)
	windex=$((windex+1))
fi

escape_squotes() { echo -n "${*//\'/\'\\\'\'}"; }
quote_args() {
	local escaped=($(escape_squotes "$@"))
	local lquoted=("${escaped[@]/#/\'}")
	echo "${lquoted[@]/%/\'}"
}

format_block() {
	append_colons() { echo -n "${@/%/\;}"; }
	set -- 'set -x' "$@" 'set +x'
	echo -n \'
	escape_squotes $(append_colons "$@")
	echo -n \'

}

format_bash() {
	echo -n "/bin/bash -c "
	format_block "$@"
}

format_cmd() {
	if [ ! -z $duration ]; then
		set -- "export TANGO_TIMEOUT=$duration" "$@"
	fi
	local cmd="$(format_bash "$@")"
	echo -n "$cmd"
	if $interactive; then
		echo -n "' exec $SHELL'"
	fi
}

create_cmd() {
	cdsrc=(
		"cd '$cwd'"
		"source '$venv/bin/activate'"
	)
	local run="./run.sh $(quote_args "$@")"
	format_cmd "${pre[@]}" "${cdsrc[@]}" "$run" "${post[@]}"
}


if [ ! ${#args[@]} -eq 0 ]; then
	if $notify; then
		channel="$sessname:$windex:0"
		post=("tmux wait -S $channel")
		echo "$channel"
	fi
	cmd="$(create_cmd "${args[@]}")"
	tmux new-window -k -t "$sessname:$windex" -n "${label:-cmd}" "$cmd"
else
	for target in "${targets[@]}"; do
		for ((i=0;i<count;i++)); do
			tdir="targets/$target"
			wdir="$outdir/$i/workdir$label"
			if [[ -e "$tdir/$wdir" ]]; then
				echo "$tdir/$wdir already exists; aborting"
				if [ "$new" = true ]; then
					tmux kill-session -t "$sessname"
				fi
				exit -1
			fi
			args=(
				-v "$tdir/fuzz.json"
				--override fuzzer.work_dir "$wdir"
			)
			for o in "${overrides[@]}"; do
				arg="$(IFS=\=; split=($o); quote_args "${split[@]}")"
				args+=("--override $arg")
			done
			pindex=$((i+windex))
			if $notify; then
				channel="$sessname:$windex:$i"
				unset post
				post=("tmux wait -S $channel")
				echo "$channel"
			fi
			cmd="$(create_cmd "${args[@]}")"
			tmux new-window -k -t "$sessname:$pindex" -n "$target$label" "$cmd"
			if [[ ! $pindex -eq $windex ]]; then
				tmux join-pane -t "$sessname:$windex" -s $pindex
				tmux select-layout -t "$sessname:$windex" tiled
			fi
		done
		if [[ $i -gt 1 ]]; then
			tmux select-layout -t "$sessname:$windex" tiled
		fi
		windex=$((windex+1))
	done
fi

if ! $detach; then
	tmux attach -t "$sessname"
fi
