import pickle
import pandas as pd
import os
import glob

def main():

    # Create a list of pickle file names
    pickles = []
    for file in glob.glob(os.path.join(os.getcwd(), "data", "output", "ner", "*.pickle")):
        pickles.append(file)

    # Load each pickle file and create the resultant CSV file
    for file in pickles:
        with open(file, 'rb') as f:
            entities = pickle.load(f)

        # Add all the names in the entity set
        entity_set = set(entities.keys())
        final_list = []
        curr_dir = os.getcwd()
        file_name_list = os.path.splitext(os.path.basename(file))[0].split('_')[2:]
        file_name = file_name_list[0]
        for string in file_name_list[1:]:
            file_name += '_' + string

        df = pd.read_csv(os.path.join(curr_dir, "data", "output", "kg", file_name + ".txt-out.csv"))

        # Parse every row present in the intermediate CSV file
        triplet = set()
        for _, row in df.iterrows():
            row.iloc[0] = row.iloc[0].strip()
            # If the entity is present in the entity set, only then parse further
            if row.iloc[0] in entity_set:
                added = False
                e2_sentence = row.iloc[2].split(' ')
                # Check every word in entity2, and add a new row triplet if it is present in entity2
                for entity in e2_sentence:
                    if entity in entity_set:
                        _ = (entities[row.iloc[0]], row.iloc[0], row.iloc[1], entities[entity], row.iloc[2])
                        triplet.add(_)
                        added = True
                if not added:
                    _ = (entities[row.iloc[0]], row.iloc[0], row.iloc[1], 'O', row.iloc[2])
                    triplet.add(_)

        # Convert the pandas DataFrame into CSV
        processed_pd = pd.DataFrame(list(triplet), columns=['Type', 'Entity 1', 'Relationship', 'Type', 'Entity2'])
        output_csv_path = os.path.join(os.getcwd(), "data", "result", os.path.splitext(os.path.basename(file))[0] + '.csv')
        processed_pd.to_csv(output_csv_path, encoding='utf-8', index=False)

        print("Processed", os.path.basename(file))

    print("Files processed and saved")

if __name__ == '__main__':
    main()
