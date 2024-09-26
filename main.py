from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Adjusted values for higher accuracy
TP = 200  # True Positives
TN = 60   # True Negatives
FP = 6   # False Positives
FN = 4    # False Negatives

# Calculate metrics
accuracy = accuracy_score([1]*TP + [0]*TN + [1]*FN + [0]*FP, [1]*TP + [1]*TN + [0]*FN + [0]*FP)
precision = precision_score([1]*TP + [0]*TN + [1]*FN + [0]*FP, [1]*TP + [1]*TN + [0]*FN + [0]*FP)
recall = recall_score([1]*TP + [0]*TN + [1]*FN + [0]*FP, [1]*TP + [1]*TN + [0]*FN + [0]*FP)
f1 = f1_score([1]*TP + [0]*TN + [1]*FN + [0]*FP, [1]*TP + [1]*TN + [0]*FN + [0]*FP)

# Print metrics
print("Confusion Matrix:")
print(confusion_matrix([1]*TP + [0]*TN + [1]*FN + [0]*FP, [1]*TP + [1]*TN + [0]*FN + [0]*FP))
print("\nAccuracy: {:.2%}".format(accuracy))
print("Precision: {:.2%}".format(precision))
print("Recall: {:.2%}".format(recall))
print("F1 Score: {:.2%}".format(f1))
