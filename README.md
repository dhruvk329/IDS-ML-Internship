# IDS Machine Learning Internship

This repo holds the main work I did during my machine learning internship at IDS InfoTech. The projects here use computer vision and natural language tools to help the publishing and editorial teams handle large volumes of documents more easily.

There are three projects in this repo.

## Document Layout Classifier (cnn_document_classifier.py)

A computer vision model built with a CNN that looks at document images and sorts them by layout type. It reached about 92% accuracy on the test set, which helped cut down the manual checking the team had to do when processing documents.

## Document Search with RAG (rag_document_search.py)

A search tool built using Retrieval-Augmented Generation. Instead of searching by exact keywords, internal teams can ask questions in plain English and get back relevant material from large client document archives. This made it faster for editors to find reference documents while working on bigger projects.

## Document Router (document_router.py)

A classifier built with Random Forest and Decision Tree models that sorts incoming documents into the right category. It reached around 88% accuracy and flags unclear cases for a person to review, which improved how documents were routed across the team's workflows.

## Built with

- Python
- TensorFlow / Keras for the CNN
- scikit-learn for the Random Forest and Decision Tree models
- Standard NLP and embedding tools for the RAG search

## Notes

This was an internship project, so parts of it were built around IDS's internal data and systems. The code here shows the approach and the models I worked on.
