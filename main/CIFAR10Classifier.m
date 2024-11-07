% Clear the workspace and command window to ensure a clean environment
clear; clc;

%% 1. Parameter Initialization
% -------------------------------------------------------------------------
% Define essential parameters for the classification task, including the
% student ID for reproducibility, number of classes to select, training and
% testing sample sizes, and the number of neighbors (K) for K-Nearest Neighbors (KNN).
% -------------------------------------------------------------------------

% Student ID used as a seed for random number generation to ensure reproducibility
studentID = 38813769;
rng(studentID, 'twister');  % Initialize the random number generator with the 'twister' algorithm

% Define the number of classes to select from the dataset
num_classes = 3;

% Define the number of samples for training and testing
num_train = 9000;
num_test = 9000;

% Define the number of nearest neighbors to consider in KNN
K = 5;

% Display the initialized parameters for verification
disp('--- Parameters Set ---');
disp(['Student ID: ', num2str(studentID)]);
disp(['Number of Classes: ', num2str(num_classes)]);
disp(['Training Samples: ', num2str(num_train)]);
disp(['Testing Samples: ', num2str(num_test)]);
disp(['K (for KNN): ', num2str(K)]);
disp(' ');

%% 2. Data Loading
% -------------------------------------------------------------------------
% Load the CIFAR-10 dataset from the provided .mat file, convert image data to
% double precision for computational purposes, and adjust label indices to
% start from 1 (MATLAB's default indexing).
% -------------------------------------------------------------------------

disp('--- Loading CIFAR-10 Data ---');

% Load the CIFAR-10 data, which includes 'data', 'labels', and 'label_names'
load('cifar-10-data.mat');

% Convert image data to double precision for numerical computations
data = double(data);

% Adjust labels to start from 1 instead of 0 to align with MATLAB's indexing
labels = labels + 1;

% 'label_names' remains unchanged as it contains categorical names for labels
label_names = label_names;

disp('Data loaded and converted to double.');
disp(' ');

%% 3. Class Selection
% -------------------------------------------------------------------------
% Randomly select a specified number of classes from the dataset using the
% student ID as the seed. Filter the dataset to include only the selected
% classes and save the selected class indices for submission.
% -------------------------------------------------------------------------

disp('--- Selecting Classes ---');

% Randomly permute integers from 1 to 10 (representing the 10 classes) and select the first 'num_classes'
classes = randperm(10, num_classes)';

% Save the selected classes to 'cw1.mat' for submission
save('cw1.mat', 'classes');

% Display the selected classes
disp(['Selected Classes: ', num2str(classes')]);

% Create a logical index to filter data belonging to the selected classes
selected = ismember(labels, classes);

% Filter the image data and labels to include only the selected classes
data = data(selected, :, :, :);
labels = labels(selected);

disp('Data filtered to selected classes.');
disp(' ');

%% 4. Label Mapping
% -------------------------------------------------------------------------
% Map the original class labels to a new set of indices starting from 1.
% This simplifies handling of labels, especially when dealing with a subset
% of classes.
% -------------------------------------------------------------------------

disp('--- Mapping Labels to Class Indices ---');

% 'unique' returns the unique class labels and maps them to new indices
[unique_classes, ~, labels_mapped] = unique(labels);

disp('Labels mapped.');
disp(' ');

%% 5. Data Splitting
% -------------------------------------------------------------------------
% Split the filtered dataset into training and testing sets using a
% pseudo-random permutation based on the student ID. Save the training indices
% for submission.
% -------------------------------------------------------------------------

disp('--- Splitting Data into Training and Testing Sets ---');

% Reinitialize the random number generator for reproducibility
rng(studentID, 'twister');

% Generate a random permutation of indices based on the number of samples
idx = randperm(size(data, 1));

% Select the first 'num_train' indices for training
train_idx = idx(1:num_train);

% Select the next 'num_test' indices for testing
test_idx = idx(num_train + 1:num_train + num_test);

% Save the training indices to 'cw1.mat' for submission
training_index = train_idx';
save('cw1.mat', 'training_index', '-append');

% Display the number of training and testing samples
disp('Data split completed.');
disp(['Number of Training Samples: ', num2str(length(train_idx))]);
disp(['Number of Testing Samples: ', num2str(length(test_idx))]);
disp(' ');

%% 6. Data Preparation
% -------------------------------------------------------------------------
% Reshape the 4D image data into 2D matrices where each row represents an
% image. This format is suitable for training and testing the classification
% models. Also, extract the corresponding labels for training and testing.
% -------------------------------------------------------------------------

disp('--- Preparing Data ---');

% Reshape the training data from 4D (samples × height × width × channels)
% to 2D (samples × features) by flattening each image into a single row
X_train = reshape(data(train_idx, :, :, :), num_train, []);

% Similarly, reshape the testing data
X_test = reshape(data(test_idx, :, :, :), num_test, []);

% Extract the mapped labels for training and testing sets
y_train = labels_mapped(train_idx);
y_test = labels_mapped(test_idx);

disp('Data reshaped for training and testing.');
disp(' ');

%% 7. K-Nearest Neighbors (KNN) with Euclidean Distance
% -------------------------------------------------------------------------
% Implement the KNN classifier using the Euclidean distance metric. Predict
% the labels for the testing data, calculate accuracy, generate a confusion
% matrix, and record the time taken for classification.
% -------------------------------------------------------------------------

disp('=== KNN with Euclidean Distance ===');

% Inform the user about the classification process
disp('Classifying test data using KNN (Euclidean distance)...');

% Call the custom KNN classifier function with 'euclidean' distance
[knnL2_pred, knnL2_timetaken] = knn_classifier(X_train, y_train, X_test, K, 'euclidean');

% Calculate accuracy as the proportion of correct predictions
knnL2_accuracy = mean(knnL2_pred == y_test);

% Generate a confusion matrix to evaluate classification performance
knnL2_confusionmatrix = confusionmat(y_test, knnL2_pred);

% Display the results of KNN with Euclidean distance
disp(['KNN (Euclidean) completed in ', num2str(knnL2_timetaken), ' seconds.']);
disp(['KNN (Euclidean) Accuracy: ', num2str(knnL2_accuracy * 100), '%']);
disp(' ');

%% 8. K-Nearest Neighbors (KNN) with Cosine Distance
% -------------------------------------------------------------------------
% Implement the KNN classifier using the Cosine distance metric. Normalize the
% training and testing data to enhance cosine similarity performance. Predict
% the labels, calculate accuracy, generate a confusion matrix, and record the
% time taken.
% -------------------------------------------------------------------------

disp('=== KNN with Cosine Distance ===');

% Inform the user about the normalization process
disp('Normalizing data for cosine distance...');

% Normalize the training data along each row (sample) to unit length
X_train_norm = normalize(X_train, 2);

% Normalize the testing data similarly
X_test_norm = normalize(X_test, 2);

% Inform the user about the classification process
disp('Classifying test data using KNN (Cosine distance)...');

% Call the custom KNN classifier function with 'cosine' distance
[knnCosine_pred, knnCosine_timetaken] = knn_classifier(X_train_norm, y_train, X_test_norm, K, 'cosine');

% Calculate accuracy as the proportion of correct predictions
knnCosine_accuracy = mean(knnCosine_pred == y_test);

% Generate a confusion matrix to evaluate classification performance
knnCosine_confusionmatrix = confusionmat(y_test, knnCosine_pred);

% Display the results of KNN with Cosine distance
disp(['KNN (Cosine) completed in ', num2str(knnCosine_timetaken), ' seconds.']);
disp(['KNN (Cosine) Accuracy: ', num2str(knnCosine_accuracy * 100), '%']);
disp(' ');

%% 9. Support Vector Machine (SVM) for Multiclass Classification
% -------------------------------------------------------------------------
% Train a Support Vector Machine (SVM) model using MATLAB's built-in
% 'fitcecoc' function, which handles multiclass classification by decomposing
% it into multiple binary classification tasks. Predict the labels, calculate
% accuracy, generate a confusion matrix, and record the time taken.
% -------------------------------------------------------------------------

disp('=== Training SVM for Multiclass Classification ===');

% Call the generic training and prediction function with 'fitcecoc' (SVM)
[SVM_pred, SVM_timetaken] = train_and_predict(@fitcecoc, X_train, y_train, X_test);

% Calculate accuracy as the proportion of correct predictions
SVM_accuracy = mean(SVM_pred == y_test);

% Generate a confusion matrix to evaluate classification performance
SVM_confusionmatrix = confusionmat(y_test, SVM_pred);

% Display the results of the SVM classifier
disp(['SVM completed in ', num2str(SVM_timetaken), ' seconds.']);
disp(['SVM Accuracy: ', num2str(SVM_accuracy * 100), '%']);
disp(' ');

%% 10. Decision Tree Classifier
% -------------------------------------------------------------------------
% Train a Decision Tree classifier using MATLAB's built-in 'fitctree'
% function. Predict the labels, calculate accuracy, generate a confusion
% matrix, and record the time taken.
% -------------------------------------------------------------------------

disp('=== Training Decision Tree Classifier ===');

% Call the generic training and prediction function with 'fitctree' (Decision Tree)
[decisiontree_pred, decisiontree_timetaken] = train_and_predict(@fitctree, X_train, y_train, X_test);

% Calculate accuracy as the proportion of correct predictions
decisiontree_accuracy = mean(decisiontree_pred == y_test);

% Generate a confusion matrix to evaluate classification performance
decisiontree_confusionmatrix = confusionmat(y_test, decisiontree_pred);

% Display the results of the Decision Tree classifier
disp(['Decision Tree completed in ', num2str(decisiontree_timetaken), ' seconds.']);
disp(['Decision Tree Accuracy: ', num2str(decisiontree_accuracy * 100), '%']);
disp(' ');

%% 11. Saving Results
% -------------------------------------------------------------------------
% Save all the required metrics (accuracy, confusion matrices, and time
% taken) for each classification model into the 'cw1.mat' file. This file
% will be submitted as part of the coursework.
% -------------------------------------------------------------------------

disp('--- Saving Results to cw1.mat ---');

% Save the metrics with appropriate variable names following the format
% "modelname_measure" (e.g., 'knnL2_accuracy', 'SVM_timetaken')
save('cw1.mat', ...
    'knnL2_accuracy', 'knnL2_confusionmatrix', 'knnL2_timetaken', ...
    'knnCosine_accuracy', 'knnCosine_confusionmatrix', 'knnCosine_timetaken', ...
    'SVM_accuracy', 'SVM_confusionmatrix', 'SVM_timetaken', ...
    'decisiontree_accuracy', 'decisiontree_confusionmatrix', 'decisiontree_timetaken', ...
    '-append');  % Use '-append' to add variables without overwriting existing data

disp('Results saved successfully.');
disp(' ');

%% 12. Image Visualization
% -------------------------------------------------------------------------
% Visualize four randomly selected images from the selected classes. The
% images are displayed in a 1-row, 4-column subplot with their respective
% class labels as titles. The figure is saved as a PNG file for inclusion
% in the report.
% -------------------------------------------------------------------------

disp('--- Visualizing Random Images from Selected Classes ---');

% Call the visualization function to display and save random images
visualize_random_images(data, labels_mapped, label_names(unique_classes));

disp('Image visualization completed. Saved as random_images.png.');
disp(' ');

%% 13. Completion Message
% -------------------------------------------------------------------------
% Display a final message indicating that all tasks have been completed
% successfully.
% -------------------------------------------------------------------------

disp('=== All tasks completed successfully. ===');

%% 14. Function Definitions
% -------------------------------------------------------------------------
% Define the custom functions used in the script: 'knn_classifier',
% 'train_and_predict', and 'visualize_random_images'.
% -------------------------------------------------------------------------

%% 14.1. KNN Classifier Function
% -------------------------------------------------------------------------
% Implements the K-Nearest Neighbors (KNN) algorithm with support for
% different distance metrics. Processes test data in batches to manage
% memory and computation efficiently. Returns the predicted labels and
% the time taken for classification.
% -------------------------------------------------------------------------
function [predictions, timetaken] = knn_classifier(X_train, y_train, X_test, K, dist)
    % Start the timer to measure the duration of classification
    tic;
    
    % Initialize a zero vector to store predictions for all test samples
    predictions = zeros(size(X_test, 1), 1);
    
    % Define the size of each batch to process test samples in manageable chunks
    batch_size = 500;
    
    % Calculate the total number of batches required
    num_batches = ceil(size(X_test, 1) / batch_size);
    
    % Loop through each batch to perform KNN classification
    for b = 1:num_batches
        % Determine the starting and ending indices for the current batch
        idx_start = (b - 1) * batch_size + 1;
        idx_end = min(b * batch_size, size(X_test, 1));
        
        % Create an index vector for the current batch
        idx = idx_start:idx_end;
        
        % Compute pairwise distances between test samples in the current batch and all training samples
        D = pdist2(X_test(idx, :), X_train, dist);
        
        % Sort the distances in ascending order and retrieve the indices of the K nearest neighbors
        [~, I] = sort(D, 2);
        
        % Assign the most frequent class label among the K nearest neighbors as the prediction
        predictions(idx) = mode(y_train(I(:, 1:K)), 2);
        
        % Display progress messages every 10 batches or at the final batch
        if mod(b, 10) == 0 || b == num_batches
            disp(['Processed batch ', num2str(b), ' of ', num2str(num_batches), '...']);
        end
    end
    
    % Stop the timer and record the total time taken for classification
    timetaken = toc;
end

%% 14.2. Train and Predict Function
% -------------------------------------------------------------------------
% A generic function to train a given classification model and make
% predictions on test data. It measures the time taken for both training
% and prediction phases.
% -------------------------------------------------------------------------
function [predictions, timetaken] = train_and_predict(model_func, X_train, y_train, X_test)
    % Start the timer to measure the duration of training and prediction
    tic;
    
    % Convert the function handle to a string for informative display messages
    model_name = func2str(model_func);
    
    % Display a message indicating the start of model training
    disp(['Training ', model_name, ' model...']);
    
    % Train the model using the provided training data and labels
    model = model_func(X_train, y_train);
    
    % Display a message indicating the completion of model training and start of prediction
    disp([model_name, ' training completed. Predicting...']);
    
    % Use the trained model to predict labels for the test data
    predictions = predict(model, X_test);
    
    % Display a message indicating the completion of prediction
    disp([model_name, ' prediction completed.']);
    
    % Stop the timer and record the total time taken for training and prediction
    timetaken = toc;
end

%% 14.3. Image Visualization Function
% -------------------------------------------------------------------------
% Displays four randomly selected images from the dataset in a 1-row,
% 4-column subplot. Each image is labeled with its corresponding class
% name. The figure is saved as 'random_images.png' for inclusion in the
% report.
% -------------------------------------------------------------------------
function visualize_random_images(data, labels, label_names)
    % Create a new figure window with normalized units and specified position
    figure('Units', 'normalized', 'Position', [0.1 0.3 0.8 0.4]);
    
    % Randomly select four unique indices from the dataset
    idx = randperm(size(data, 1), 4);
    
    % Loop through each selected index to display the corresponding image
    for i = 1:4
        % Create a subplot in a 1-row, 4-column grid
        subplot(1, 4, i);
        
        % Extract the image data for the current index, squeeze to remove singleton dimensions,
        % and normalize pixel values to the range [0, 1] for display
        img = squeeze(data(idx(i), :, :, :)) / 255;
        
        % Display the image using imshow for accurate rendering
        imshow(img);
        
        % Set the title of the subplot to the corresponding class label with enhanced font properties
        title(label_names{labels(idx(i))}, 'FontSize', 12, 'FontWeight', 'bold');
    end
    
    % Add a super title to the entire figure for context
    sgtitle('Random Images from Selected Classes', 'FontSize', 16, 'FontWeight', 'bold');
    
    % Save the figure as a PNG file in the current working directory
    saveas(gcf, 'random_images.png');
end
