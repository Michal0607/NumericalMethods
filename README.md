# Zastosowanie PCA i SVD w Uczeniu Maszynowym

## Wprowadzenie

Ten projekt koncentruje się na zastosowaniu i porównaniu dwóch technik redukcji wymiarowości - Analizy Głównych Składowych (PCA) oraz Rozkładu na Wartości Osobliwe (SVD) - w celu poprawy wydajności i efektywności algorytmów uczenia maszynowego. Projekt został zaimplementowany w Pythonie, z wykorzystaniem bibliotek takich jak NumPy, pandas i scikit-learn.

## Opis technologiczny

Projekt został zrealizowany przy użyciu języka programowania Python i obejmuje kilka skryptów, które implementują PCA i SVD na różnych zestawach danych. Skrypty są zaprojektowane tak, aby były łatwe w użyciu i modyfikacji.

### Skrypty

- `PCA` i `SVD` implementacje na zestawach danych:
  - Iris (`pca_iris.py`, `svd_iris.py`)
  - Breast Cancer (`pca_bc.py`, `svd_bc.py`,`LR_PCA.py`, `LR_SVD.py`)
  - Titanic (`pca_titanic.py`, `svd_titanic.py`)
  - Heloc (`heloc_pca.py`)
  
Każdy skrypt wykonuje analizę redukcji wymiarowości na wybranym zestawie danych, z wynikami wizualizacji włączonymi do analizy.

## Instalacja

Aby uruchomić projekt lokalnie, wykonaj:

```bash
git clone https://github.com/Michal0607/NumericalMethods.git
```

## Użycie

Aby uruchomić analizę PCA lub SVD na wybranym zbiorze danych, uruchom odpowiedni skrypt Python:

```bash
python pca_iris.py
python svd_iris.py
```
Podobne polecenia należy użyć dla innych zbiorów danych zgodnie z nazwami plików.

## Wyniki

Wyniki projektu zawierają wizualizacje danych przed i po zastosowaniu PCA oraz SVD, co pozwala na ocenę skuteczności tych metod w redukcji wymiarów i poprawie klarowności danych.

## Wnioski

Projekt pokazuje, że zarówno PCA, jak i SVD są efektywnymi metodami redukcji wymiarów dla różnorodnych typów danych. Wyniki sugerują, że odpowiedni dobór metody zależy od specyfiki danych i wymagań analizy.

## Autorzy 

Michał Szulierz, Filip Trochimiuk
