# Directory containing the .tar.gz files
SOURCE_DIR="."
# Directory where all extracted files will be merged
DEST_DIR="gs_data"

# Create the destination directory if it doesn't exist
mkdir -p "$DEST_DIR"


for file in "$SOURCE_DIR"/*.zip; 
do
  # Extract the .tar.gz file
  unzip "$file" -d "$DEST_DIR"

done

mkdir "$DEST_DIR"/03001627
mv "$DEST_DIR"/03001627_0/* "$DEST_DIR"/03001627
mv "$DEST_DIR"/03001627_1/* "$DEST_DIR"/03001627

rm -r "$DEST_DIR"/03001627_0
rm -r "$DEST_DIR"/03001627_1


mkdir "$DEST_DIR"/04379243
mv "$DEST_DIR"/04379243_0/* "$DEST_DIR"/04379243
mv "$DEST_DIR"/04379243_1/* "$DEST_DIR"/04379243

rm -r "$DEST_DIR"/04379243_0
rm -r "$DEST_DIR"/04379243_1


