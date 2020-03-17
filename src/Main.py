from src.DataExtractor import DataExtractor
from src.DataFilter import DataFilter
from src.DataParser import DataParser

raw_data = DataParser.parse()
filtered_data = DataFilter.filter(raw_data)
DataExtractor.extract(filtered_data)
