# ViCE: Visual Counterfactual Explanations for Machine Learning Models


[Oscar Gomez](https://github.com/oscargomezq/), [Steffen Holter](https://github.com/5teffen/), [Jun Yuan](https://github.com/junyuanjun/) and [Enrico Bertini](http://enrico.bertini.io/)

The continued improvements in the predictive accuracy of machine learning models have allowed for their widespread practical application. Yet, many decisions made with seemingly accurate models still require verification by domain experts. In addition, end-users of a model also want to understand the reasons behind specific decisions. Thus, the need for interpretability is increasingly paramount. In this paper we present an interactive visual analytics tool, ViCE, that generates counterfactual explanations to contextualize and evaluate model decisions. Each sample is assessed to identify the minimal set of changes needed to flip the model's output. These explanations aim to provide end-users with personalized actionable insights with which to understand, and possibly contest or improve, automated decisions. The results are effectively displayed in a visual interface where counterfactual explanations are highlighted and interactive methods are provided for users to explore the data and model. The functionality of the tool is demonstrated by its application to a home equity line of credit dataset.


ViCE requires the following packages:

* numpy
* pandas
* scikit-learn
* pickle


Getting started
-------------------------
```python
import vice

# Initialize own local dataset
path = "~/diabetes.csv"
d = vice.Data(path)

# Initialize example dataset
#   "diabetes": diabetes dataset
#   "grad": graduate admissions dataset

d = vice.Data(example = "grad")
```
Advanced parameters for custom dataset initialization include
* :param data: preloaded data array with first row as feature names
* :param target: target column index (by default -1)
* :param exception: non-actionable feature column index list
* :param categorical: categorical feature column index list


```python
# Initialize pre-trained model
m = vice.Model(model, backend = "scikit")

# Load model from path
model_path = "~/diabetes_svm.sav"
m = vice.Model().load_model(model_path)


# Train a model using initialized dataset
d = vice.Data(example = "diabetes")
m = vice.Model().train_model(d, type='svm')
```

Model allows backend specification ("scikit", coming soon: "tf", "pytorch"). 
Various model types available for immediate training in real-time
* "svm": support vector machines
* "rf": random forests
* "lg": logistic regression


```python
# Initialize basic counter factual explanation visualiser
vice_explainer = vice.Vice(d,m, mode = "basic")

# For more a more comprehensive tool select "full" mode. Note: this option can be time intensive
vice_explainer = vice.Vice(d,m, mode = "full")

```
To get a general feel for the ViCE tool it is sufficient to use "basic" mode. It is optimal for creating explanations with a few clicks. On the other hand, "full" mode uses data preprocessing to contextualize the individual explanations with regards to the rest of the samples in the dataset. For very large datasets, this can take a considerable amount of time, thus such data should ideally be sampled. (Coming soon: in built sampling)

Generating Explanations
-------------------------
Vice generates a visual explanation for the specified sample. 

```python

vice_explainer.generate_explanation(sample_no = 1)

```






