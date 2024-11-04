MAX_RETRIES=${MAX_RETRIES:-6}
SLEEP_SECOND=${SLEEP_SECOND:-10}
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
  if brsmi | grep -q "No running process found"; then
    echo "No running process found on BR GPUs, free to use!"
    break
  else
    echo "[${RETRY_COUNT}/${MAX_RETRIES}] BR GPUs are occupied by others, retry after ${SLEEP_SECOND} seconds..."
    sleep $SLEEP_SECOND
    RETRY_COUNT=$((RETRY_COUNT + 1))
  fi
done

if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
  echo "BR GPUs are occupied by others, please wait until free!!!"
  exit -1
fi