#!/usr/bin/env bash
# Compares v1 (cgo, libwebp) with v2 (pure Go, and the assembly kernels on
# arm64 and amd64)
# on the same workloads; see the Benchmarks section of ../README.md.
#
#   benchmarks/run.sh               # everything (about an hour)
#   ROUNDS=3 RSS_ROUNDS=1 SAT=0 benchmarks/run.sh    # a quick look
#
# Environment:
#   BUILDS      builds to compare, in table order (default "v1 v2-purego v2"):
#               v1 = github.com/kolesa-team/go-webp v1.0.5 + the system
#               libwebp (needs cgo and libwebp's pkg-config file);
#               v2-purego = this repository built with -tags purego (the
#               code other architectures run);
#               v2 = this repository as built by default.
#   ROUNDS      benchmark rounds (default 10); the builds alternate within
#               every round, in a rotating order, so that a change of the
#               machine's state affects all of them alike
#   RSS_ROUNDS  peak-memory rounds (default 3; 0 skips them)
#   SAT         1 (default) also runs the saturation benchmarks
#               (GOMAXPROCS calls at a time), 0 skips them
#   OUT         output directory (default benchmarks/out)
#
# Results: $OUT/benchstat.txt (medians and comparisons, the first build is
# the reference), $OUT/rss.txt (peak resident memory, medians), raw
# outputs in $OUT/raw/, the machine description in $OUT/env.txt.
set -euo pipefail

cd "$(dirname "$0")"
BUILDS=${BUILDS:-"v1 v2-purego v2"}
ROUNDS=${ROUNDS:-10}
RSS_ROUNDS=${RSS_ROUNDS:-3}
SAT=${SAT:-1}
OUT=${OUT:-out}
mkdir -p "$OUT/bin" "$OUT/raw"
OUT=$(cd "$OUT" && pwd)
BIN=$OUT/bin
RAW=$OUT/raw
DATA=$(cd ../test_data/images && pwd)

BENCHSTAT=${BENCHSTAT:-$(command -v benchstat || echo "$(go env GOPATH)/bin/benchstat")}
if [ ! -x "$BENCHSTAT" ]; then
	echo "benchstat not found: go install golang.org/x/perf/cmd/benchstat@latest" >&2
	exit 1
fi

load() { uptime | sed 's/.*load averages*: *//'; }

{
	echo "date: $(date)"
	echo "go: $(go version)"
	echo "os: $(uname -srm)"
	case "$(uname -s)" in
	Darwin)
		echo "cpu: $(sysctl -n machdep.cpu.brand_string), $(sysctl -n hw.ncpu) cores, $(($(sysctl -n hw.memsize) / 1073741824)) GiB"
		echo "macos: $(sw_vers -productVersion)"
		;;
	Linux)
		echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //'), $(nproc) cores"
		;;
	esac
	echo "libwebp (v1): $(pkg-config --modversion libwebp 2>/dev/null || echo unknown)"
	echo "builds: $BUILDS; rounds: $ROUNDS; rss rounds: $RSS_ROUNDS; saturation: $SAT"
	echo "load at start: $(load)"
} >"$OUT/env.txt"
cat "$OUT/env.txt"

echo "building..."
for b in $BUILDS; do
	case $b in
	v1) dir=./v1 tags= ;;
	v2-purego) dir=./v2 tags=purego ;;
	v2) dir=./v2 tags= ;;
	*)
		echo "unknown build $b" >&2
		exit 1
		;;
	esac
	if ! go test -c -tags "$tags" -o "$BIN/$b.test" "$dir" 2>"$RAW/build-$b.log" ||
		! go build -tags "$tags" -o "$BIN/rss-$b" "$dir/rss" 2>>"$RAW/build-$b.log"; then
		cat "$RAW/build-$b.log" >&2
		[ "$b" = v1 ] && echo "v1 needs cgo and libwebp (brew install webp / apt install libwebp-dev pkg-config), or run with BUILDS=\"v2-purego v2\"" >&2
		exit 1
	fi
done

if [ "$ROUNDS" -gt 0 ]; then
	set -- $BUILDS
	n=$#
	for b in $BUILDS; do : >"$RAW/$b.txt"; : >"$RAW/sat-$b.txt"; done
	for r in $(seq 1 "$ROUNDS"); do
		for k in $(seq 0 $((n - 1))); do
			i=$(((k + r) % n + 1))
			b=${!i}
			echo "round $r/$ROUNDS $b (load $(load))"
			"$BIN/$b.test" -test.run '^$' -test.bench '^BenchmarkDecode$' -test.benchtime 1s -test.benchmem -test.count 1 >>"$RAW/$b.txt"
			"$BIN/$b.test" -test.run '^$' -test.bench '^BenchmarkEncode$' -test.benchtime 2s -test.benchmem -test.count 1 >>"$RAW/$b.txt"
			if [ "$SAT" = 1 ]; then
				"$BIN/$b.test" -test.run '^$' -test.bench '^BenchmarkSat' -test.benchtime 2s -test.benchmem -test.count 1 >>"$RAW/sat-$b.txt"
			fi
		done
	done
	files=()
	satfiles=()
	for b in $BUILDS; do
		files+=("$RAW/$b.txt")
		satfiles+=("$RAW/sat-$b.txt")
	done
	(cd "$RAW" && "$BENCHSTAT" -ignore pkg "${files[@]##*/}") >"$OUT/benchstat.txt"
	if [ "$SAT" = 1 ]; then
		(cd "$RAW" && "$BENCHSTAT" -ignore pkg "${satfiles[@]##*/}") >"$OUT/benchstat-saturation.txt"
	fi
fi

if [ "$RSS_ROUNDS" -gt 0 ]; then
	LL12=$OUT/ll12.webp
	last=${BUILDS##* }
	[ -s "$LL12" ] || "$BIN/rss-$last" -mode gen12ll -out "$LL12"
	: >"$RAW/rss.log"
	rss() {
		local b=$1 w=$2
		shift 2
		printf '%s\t%s\t%s\n' "$b" "$w" "$("$@" | tail -1)" >>"$RAW/rss.log"
	}
	for r in $(seq 1 "$RSS_ROUNDS"); do
		for b in $BUILDS; do
			echo "rss round $r/$RSS_ROUNDS $b (load $(load))"
			t=$BIN/rss-$b
			rss "$b" "enc12-lossy" "$t" -mode enc12
			rss "$b" "enc12-lossy-threads" "$t" -mode enc12 -threads
			rss "$b" "enc12-lossless-x1" "$t" -mode enc12ll -r 1
			rss "$b" "enc12-lossless-x3" "$t" -mode enc12ll -r 3
			rss "$b" "enc12-lossless-x3-gomemlimit450" env GOMEMLIMIT=450MiB "$t" -mode enc12ll -r 3
			rss "$b" "dec12-lossless" "$t" -mode dec12 -file "$LL12"
			rss "$b" "conc15x4-lossless-logo" "$t" -mode conc -r 4 -file "$DATA/webp-logo-lossless.webp"
			rss "$b" "conc15x4-lossy-alpha-logo" "$t" -mode conc -r 4 -file "$DATA/webp-logo-lossy.webp"
			if [ "$b" != v1 ]; then
				rss "$b" "conc15x4-lossless-logo-into" "$t" -mode conc -r 4 -into -file "$DATA/webp-logo-lossless.webp"
				rss "$b" "conc15x4-lossy-alpha-logo-into" "$t" -mode conc -r 4 -into -file "$DATA/webp-logo-lossy.webp"
			fi
		done
	done
	# Medians per build and workload (peak MB with min-max, wall ms).
	awk -F'\t' '{
		peak = wall = ""
		n = split($3, kv, " ")
		for (i = 1; i <= n; i++) {
			split(kv[i], p, "=")
			if (p[1] == "peak_mb") peak = p[2]
			if (p[1] == "wall_ms") wall = p[2]
		}
		print $1 "\t" $2 "\t" peak "\t" wall
	}' "$RAW/rss.log" | sort -t "$(printf '\t')" -k1,1 -k2,2 -k3,3n | awk -F'\t' '
	function flush() {
		if (key == "") return
		printf "%-10s %-34s peak %7.1f MB (%.1f-%.1f)  wall %6.0f ms  (%d runs)\n", b, w, v[int((c + 1) / 2)], v[1], v[c], wl[int((c + 1) / 2)], c
	}
	{
		k = $1 "\t" $2
		if (k != key) { flush(); key = k; b = $1; w = $2; c = 0 }
		c++; v[c] = $3; wl[c] = $4
	}
	END { flush() }' >"$OUT/rss.txt"
fi

echo "load at end: $(load)" >>"$OUT/env.txt"
echo
[ -f "$OUT/benchstat.txt" ] && cat "$OUT/benchstat.txt"
[ -f "$OUT/rss.txt" ] && cat "$OUT/rss.txt"
echo "results in $OUT"
