unzip SentimentAnalysisData.zip
%open the csv file containing the data into a table. 
T = readtable('SentimentAnalysisData\text_emotion_data.csv');

%convert the table column to an array
sentences = table2array(T(:, 2));

%build the labels vector
labels = unique(table2array(T(:, 1)));

%tokenize the sentences
documents = tokenizedDocument(sentences);

%building the Bag-Of-Words 
bag = bagOfWords(documents);

%removing common stop words and 
% words with fewer than 100 occurences
newBag = removeInfrequentWords(removeWords(bag, stopWords), 99);

%Building the full TF-IDF matrix for the resulting bag.
M = tfidf(newBag);
M1 = array2table(full(M));

%creating the features matrix for training
training_features = M1(1:6921, :);
testing_features = M1(6921:end, :);

%creating the labels vector
training_labels = T(1:6921, 1);
testing_labels = table2array(T(6921:end, 1));


%training the machine with K-Nearest Neighbour algorithm
knnmodel = fitcknn(training_features, training_labels);
knn_predictions = predict(knnmodel, testing_features);

%evaluating the model by calculating the accuracy
num_correct_predictions_knn = sum(strcmp(knn_predictions, testing_labels));
knn_accuracy = num_correct_predictions_knn./numel(testing_labels);

%visualising the model using confusion chart
figure(1)
confusionchart(testing_labels, knn_predictions, 'Title', 'K Nearest Neighbor confusion chart');

%training the machine with Naive Bayes algorithm
nbmodel = fitcnb(training_features, training_labels);
nb_predictions = predict(nbmodel, testing_features);

%evaluating the model by calculating the accuracy
num_correct_predictions_nb = sum(strcmp(nb_predictions, testing_labels));
nb_accuracy = num_correct_predictions_nb./numel(testing_labels);

%visualising the model using confusion chart
figure(2)
confusionchart(testing_labels, nb_predictions, 'Title', 'Naive Bayes confusion chart');

%training the machine with Decision Tree algorithm
treemodel = fitctree(training_features, training_labels);
tree_predictions = predict(treemodel, testing_features);

%evaluating the model by calculating the accuracy
num_correct_predictions_tree = sum(strcmp(tree_predictions, testing_labels));
tree_accuracy = num_correct_predictions_tree./numel(testing_labels);

%visualising the model using confusion chart
figure(3)
confusionchart(testing_labels, tree_predictions, 'Title', 'Decision Tree confusion chart');