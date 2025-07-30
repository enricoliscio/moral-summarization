import pandas as pd
import os
import random
import string


def generate_random_string(length=4):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def create_expert_task(assignment, summaries, base_dir):
    # Dictionary to keep track of the mappings
    mappings = pd.DataFrame(columns=['article_code', 'vanilla', 'simple', 'cot', 'oracle', 'class'], dtype=str)

    # Loop through each row in the assignment DataFrame
    for expert, row in assignment.iterrows():
        # Create a folder for the expert
        expert_dir = os.path.join(base_dir, expert)
        os.makedirs(expert_dir, exist_ok=True)
        
        # Loop through each column in the row (each column corresponds to an article)
        for article_id in row:
            # create a new empty row in mappings with index as the article_id
            mappings.loc[article_id] = None

            # Generate a random string for the article
            anonymized_article_id = generate_random_string()
            mappings.at[article_id, 'article_code'] = anonymized_article_id

            # Create a folder for the article
            article_dir = os.path.join(expert_dir, anonymized_article_id)
            os.makedirs(article_dir, exist_ok=True)
            
            # Retrieve the corresponding article row from the summaries DataFrame
            article_row = summaries.loc[summaries.index == article_id]

            scores = pd.DataFrame(columns=['score'])

            # create a folder named "summaries"
            summaries_dir = os.path.join(article_dir, 'summaries')
            os.makedirs(summaries_dir, exist_ok=True)
            
            # Loop through each column in the article row and create a txt file with the content
            for col in article_row.columns:
                if col == 'dataset':
                    continue

                content = article_row[col].values[0]
                if col == 'article_text':
                    file_name = 'article'
                    file_path = os.path.join(article_dir, f"{file_name}.txt")
                else:
                    # Generate a random string for the summary
                    file_name = generate_random_string()
                    mappings.at[article_id, col] = file_name
                    file_path = os.path.join(summaries_dir, f"{file_name}.txt")
                    # Add an empty row in scores
                    scores.loc[file_name] = None

                with open(file_path, 'w') as file:
                    file.write(str(content))

            # Save the scores to a CSV file
            scores.to_csv(os.path.join(article_dir, 'scores.csv'))

            # Add the files where they can add the justification of the differences between summaries
            explanations = pd.DataFrame(columns=['justification'])
            example_1 = f"{mappings.at[article_id, 'vanilla']}-{mappings.at[article_id, 'simple']}"
            example_2 = f"{mappings.at[article_id, 'vanilla']}-{mappings.at[article_id, 'cot']}"
            explanations.loc[example_1] = f"Enter the justification of the difference between {mappings.at[article_id, 'vanilla']} and {mappings.at[article_id, 'simple']} in this cell"
            explanations.loc[example_2] = f"Enter the justification of the difference between {mappings.at[article_id, 'vanilla']} and {mappings.at[article_id, 'cot']} in this cell"

            explanations.to_csv(os.path.join(article_dir, 'differences_justifications.csv'))

    # Save the mappings to a CSV file
    mappings_df = pd.DataFrame(mappings)
    mappings_df.to_csv(os.path.join(base_dir, 'anonymized_mappings.csv'))