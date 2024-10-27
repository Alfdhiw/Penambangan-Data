# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
dataset = pd.read_csv('tingkat-polusi-pabrik-industricandi-semarang-2020-2022.csv', sep=';')
X = dataset.iloc[:, [1, 3]].values
y = dataset.iloc[:, -1].values    # Hanya kolom label/target

# %%
print(X)

# %%
# Encode label kategori ke angka
from sklearn.preprocessing import StandardScaler, LabelEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)  # Mengonversi kategori menjadi angka

# %%
# Menampilkan label yang diwakili oleh angka setelah encoding
for index, label in enumerate(labelencoder.classes_):
    print(f"{index}: {label}")

# %%
print(y)

# %%
#membagi data menjadi training set dan test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# %%
print (X_train)

# %%
print (X_test)

# %%
print (y_train)

# %%
print (y_test)

# %%
#melakukan standarisasi karena beberapa algoritma machine learning bekerja lebih baik
#ketika data berada pada skala yang seragam
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# %%
print (X_train)

# %%
print (X_test)

# %%
#menerapkan algoritma Naïve Bayes dengan varian Gaussian Naïve Bayes untuk melakukan klasifikasi
#pada dataset yang sudah dipisahkan menjadi data training.
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# %%
#untuk memprediksi label atau kategori dari data X_test (data uji/test set) setelah model
#Gaussian Naïve Bayes selesai dilatih
y_pred = classifier.predict(X_test)

# %%
#menampilkan confusion matrix yang merupakan salah satu alat evaluasi dalam klasifikasi untuk
#memahami performa model dengan membandingkan prediksi dengan nilai sebenarnya
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# %%
#memvisualisasikan klasifier naive bayes ke dalam ruang fitur dua dimensi
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('#F1F0E8', '#739072')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('#B06161','#3A4D39', '#e7f69e'))(i), label = j)
plt.title('Naïve Bayes (Training Set)')
plt.xlabel('PM2.5')
plt.ylabel('SO2')
plt.legend()
plt.show()


