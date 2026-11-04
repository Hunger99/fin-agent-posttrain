# Machine settings (environment_installation/local.env) and the repository's output locations; needs ROOT, which each
# script computes from its own location. POSIX sh: the servers started with sh source it too. Sourced repeatedly (each
# library computes ROOT itself); every value is the same each time.
[ -n "$ROOT" ] || { echo "paths.sh: ROOT is not set" >&2; exit 1; }
[ -f "$ROOT/environment_installation/local.env" ] || { echo "paths.sh: no $ROOT/environment_installation/local.env (copy local.env.example)" >&2; exit 1; }
. "$ROOT/environment_installation/local.env"
# State and outputs of the RL chain: the strict regrades, the chain's stdout and judge state, DRY output.
STRICT_DIR=$ROOT/outputs/archive/runs/strict-regrade--hardset_v1--20261102T2246Z
CHAIN_DIR=$ROOT/outputs/archive/runs/rl-chain-c--market_v1--20261103T0726Z
DRY_DIR=$ROOT/outputs/archive/runs/rl-chain-c-dry--no-data--20261103T0825Z
