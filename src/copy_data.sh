#! /bin/bash

echo "Starting: Copy images"
cp -r /data/ocean_images /safe_outputs/ocean_images
echo "Finished. Starting: Copy annotations"
cp /data/ocean_data.json /safe_outputs/ocean_data.json
echo "Finished."
