In this project, the Adult (Census Income) dataset was used to build a logistic regression model to predict whether a person earns more than $50,000 per year. You can find out more information at https://archive.ics.uci.edu/dataset/2/adult.
The data was cleaned by removing duplicates and rows with missing values. Categorical features were transformed using one-hot encoding to make them suitable for modeling. 
The dataset was then split into training and testing sets, and standardized using StandardScaler to ensure consistent feature scaling.

![image](https://github.com/user-attachments/assets/0c8849f9-1655-4896-8a2d-717e18be0998)

After training a logistic regression model with a maximum of 200 iterations, I evaluated its performance using accuracy and a confusion matrix. 
The model achieved an accuracy of 85.14%, indicating strong overall performance. However, the confusion matrix provided deeper insight into how the model performed 
across the two income classes.

Specifically, the model correctly identified 6,313 individuals earning ≤$50K and 1,385 individuals earning >$50K. However, it also misclassified 436 low-income 
individuals as high-income (false positives) and 908 high-income individuals as low-income (false negatives). This imbalance shows that the model is more effective 
at detecting lower-income individuals than higher-income ones. The relatively high number of false negatives (high earners incorrectly predicted as low earners) may 
point to a class imbalance or feature overlap that the model struggled with.

In conclusion, while the logistic regression model performed well overall, the confusion matrix highlighted areas for improvement, especially in accurately identifying 
high-income individuals. 
