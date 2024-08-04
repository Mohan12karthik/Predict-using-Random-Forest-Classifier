# Predict Apartment Rent Using Random Forest Classifier

## Project Description
This project implements a machine learning pipeline to predict apartment rent levels (high or low) based on various features such as the apartment's age, distance to the nearest commute station, number of nearby stores, and location (latitude and longitude). Using a Random Forest Classifier, the model achieves a validation accuracy of 79.69%. The final predictions on the test dataset are exported to a CSV file for submission.

## Data Description
### Columns
- **id**: Unique ID corresponding to the apartment
- **age**: Age of the apartment
- **commute_distance**: Distance of the nearest commute station from the apartment
- **num_stores**: Number of stores near the apartment
- **location**: Location of the locality represented as latitude and longitude separated by a delimiter
- **outcome**: The predicted rent of the apartment (1 - High, 0 - Low)

## Outcome of the Result
In this data analytics project, the objective was to predict whether an apartment's rent is high or low based on several features. The dataset contained information such as the apartment's age, commute distance, number of nearby stores, and geographic location. The primary steps included data preprocessing, handling missing values, feature scaling, and building a machine learning model.

### Key Steps:
1. **Data Cleaning**: Converted age and commute distance to numeric values and extracted latitude and longitude from the location string.
2. **Imputation**: Filled missing values using the mean strategy.
3. **Feature Scaling**: Applied standard scaling to normalize the features.
4. **Model Training**: Trained a Random Forest Classifier to predict the rent outcome.

The model achieved a validation accuracy of approximately 79.69%, indicating a good fit for the data. The final step involved generating predictions for the test dataset, which were then saved into a CSV file named `submissions.csv`.

### Final Output
The output of this project is a CSV file containing the predicted rent outcomes for each apartment in the test dataset, providing a clear binary indication (1 for high rent, 0 for low rent) based on the input features. This project showcases a complete machine learning pipeline, from data preprocessing to model evaluation and prediction, useful for similar classification tasks in real estate and other domains.

## Files
- `train.csv`: Training dataset
- `test.csv`: Test dataset
- `submissions.csv`: Predictions for the test dataset
- `notebook.ipynb`: Jupyter notebook containing the code

## Usage
1. Clone the repository:
   ```bash
   git clone <repository_url>
   
2. Navigate to the project directory:
    ```bash
    cd <project-directory>
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the Jupyter notebook or Python script to preprocess the data, train the model, and generate predictions.

5. The final predictions will be saved in a file named `submissions.csv`.

## Requirements
- pandas
- numpy
- scikit-learn
