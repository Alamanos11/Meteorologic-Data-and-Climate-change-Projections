# Meteorologic-Data-and-Climate-change-Projections
Historic Meteorological Data Processing and Visualization, accessing Climate Change Projections, Bias Correction and Performance Statistics
###
This model shows a complete process of working with historic climate data and climate change projections, with several data processing steps that are very often required. These include: 
###
1)	Download historic meteorological data,
2)	Fill missing data values using a simple technique (the mean of the available data for the same months),
3)	Convert the data in different units (in this case from inches to mm of rainfall, and from Farheneit into oC of temperature),
4)	Show different types of plots to visualize the time-series of meteorological data,
5)	Download climate change projections for a specific location, from selected Global Circulation Models (GCMs), for different RCP scenarios, 
6)	Make necessary operations with the data (e.g. make the daily values of rainfall or temperature into monthly values), and visualize the output data,
7)	Test different methods for Bias Correction (e.g. Delta Change Method, Multiplicative Scaling Method, Quantile Mapping, Climate-Fit “CF” method) and check their performances with statistics.
###
Although these tasks are pretty common, the tools to perform these analyses are not always accessible, or they are often split in parts. 
In this example, a single model executing the above tasks is developed.
###
A CSV file with indicative input data is provided, along with the Python script and the Supplementary Information explaining these tasks.
###
#
Reference:
###
Alamanos, A. (2023). Historic Meteorological Data Processing and Visualization, accessing Climate Change Projections, Bias Correction and Performance Statistics. DOI: 10.13140/RG.2.2.16010.03528. Available at: https://github.com/Alamanos11/Meteorologic-Data-and-Climate-change-Projections  
