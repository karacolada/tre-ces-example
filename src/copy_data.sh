#! /bin/bash

echo "Starting: Copy images"
cp -r /scratch/ocean_images /safe_outputs/ocean_images
echo "Finished. Starting: Copy annotations"
cp /scratch/ocean_data.json /safe_outputs/ocean_data.json
echo "Finished."
