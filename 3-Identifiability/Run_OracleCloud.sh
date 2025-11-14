#!/bin/bash

# Define variables
PYTHON_SCRIPT="Likelihood_Profile_sse.py"
LOG_FILE="Results/script_output.log"
EMAIL="23640847@sun.ac.za"
TOPIC_OCID=ocid1.onstopic.oc1.af-johannesburg-1.amaaaaaajowclpaao6vhio4oahtrkrtnbiyi2jjvkengqow4vrcphkfxl34a

# Record the start time
START_TIME=$(date)
# Start counting time
SECONDS=0
# Log the start time
echo "Script $PYTHON_SCRIPT started at: $START_TIME" > $LOG_FILE

# Run the python script
python3 $PYTHON_SCRIPT >> $LOG_FILE 2>&1 &
# Get the process ID of the script
PID=$!
# Wait for the process to complete
wait $PID

# Record the end time
END_TIME=$(date)
# Calculate total running time (in seconds)
ELAPSED_TIME=$SECONDS
# Convert to more readable format (hours, minutes, seconds) and handle > 99 hours
HOURS=$(($ELAPSED_TIME/3600))
MINUTES=$(($ELAPSED_TIME%3600/60))
SECONDS_LEFT=$(($ELAPSED_TIME%60))
RUN_TIME=$(printf '%dh:%02dm:%02ds\n' $HOURS $MINUTES $SECONDS_LEFT)
# Log the end time and total running time
echo "Script $PYTHON_SCRIPT finished at: $END_TIME" >> $LOG_FILE
echo "Total running time: $RUN_TIME" >> $LOG_FILE

# Publish message
oci ons message publish --body "The Python script $PYTHON_SCRIPT has completed running on your Oracle Cloud instance.  Start time: $START_TIME.  End time: $END_TIME.  Total running time: $RUN_TIME.  Output is saved in $LOG_FILE." --topic-id $TOPIC_OCID
