SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# flatc -o $SCRIPT_DIR/core_cpp/include --cpp schemas/OptionData.fbs # cmake takes care of this
flatc -o $SCRIPT_DIR/conn_py --python schemas/OptionData.fbs

echo ">>> successfully compiled flatbuffers"