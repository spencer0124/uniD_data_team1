# run.sh
#!/bin/bash
# echo "Starting training..."
# python train.py > training_log.txt 2>&1

echo "Starting testing..."
python test.py > testing_log.txt 2>&1

echo "Process completed. Check training_log.txt and testing_log.txt for results."