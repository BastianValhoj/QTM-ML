#!/bin/bash

module load Siesta

# --- default values ---

NW_PARAM=10
NP_PARAM=4


# --- Parse Flags ---
while getopts "n:w:" opt; do
	case $opt in 
		n) NP_PARAM=$OPTARG ;;
		w) NW_PARAM=$OPTARG ;;
		\?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
	esac
done


echo "-----------------------------------"
echo "Machine: $(hostname)"
echo "Staring workflow for Nw = $NW_PARAM"
echo "Resources: $NP_PARAM processors"
echo "-----------------------------------"

# 1. Run the python script and wait for it to finish succesfully
echo "Starting step 1: tbt_edge.py..."
uv run tbt_edge.py $NW_PARAM

# Check if step 1 failed
if [ $? -ne 0 ]; then
	echo "Error: tbt_edge.py failed. Aborting."
	exit 1
fi

ARM_DIR="TBT-Nw$NW_PARAM-armchair"
ZIG_DIR="TBT-Nw$NW_PARAM-zigzag"
# Check if python actually created the directories before startin TBTrans
if [ ! -d "$ARM_DIR" ] || [ ! -d "$ZIG_DIR" ]; then
	echo "Error: Directories for Nw $NW_PARAM not found. Check tbt_edge.py output."
	exit 1
fi

echo "Step 1 complete. Starting TBTrans calculations in parallel..."

# 2. Start the armchair calculations in the background
echo "Removing old *.TBT.nc files"
rm -f $ARM_DIR/*.TBT.nc $ARM_DIR/*.SE.nc $ZIG_DIR/*.TBT.nc $ZIG_DIR/*.SE.nc

echo "-------------------------------------------------"
echo "## Check progress with: tail -f <armchair|zigzag>.Nw$NW_PARAM.log"
echo "-------------------------------------------------"

echo "Launching Armchair"
mpirun -np $NP_PARAM tbtrans < $ARM_DIR/RUNTBT.fdf > armchair.Nw$NW_PARAM.log 2>&1
#PID_ARM=$!
echo "Armchair calculations finished."

#sleep 2

# 3. Start the zig-zag calculations in the background
echo "Launching Zig-zag"
mpirun -np $NP_PARAM tbtrans < $ZIG_DIR/RUNTBT.fdf > zigzag.Nw$NW_PARAM.log 2>&1
#PID_ZIG=$!
echo "Zig-zag calculations finished."

#echo "Both jobs are running in the background. Tracking PIDs: $PID_ARM, $PID_ZIG"

# 4. Wait for both background processes to finish
#echo "Waiting for parallel tasks to complete..."

#wait $PID_ARM
#echo "Armchair calculations finished."

#wait $PID_ZIG
#echo "Zig-zag calculations finished."

echo "All calculations finished."
