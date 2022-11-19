# Art Gallery Location Analysis
###### Created by Jaime Luna, Nicholas Stein
The data for this analysis will be collected using publicly available datasets. The datasets will be obtained from census data, Foursquare API, and Satellite imagery.

With this data, we will identify optimal locations to place art galleries with respect to revenue potential while considering operating costs. Essentially, this analysis is creating a neighborhood taxonomy. Each neighborhood will be individually assessed and ranked.

In order to accomplish this analysis, we will be utilizing k-means to classify neighborhood groups. Additionally, the random forest classifier will be used to generate descriptive statistics and feature importance.

Our performance will be measured by comparing our results to actual locations of art galleries in New York. Ideally, the art gallery locations in New York will match the suggested locations as output by this project.
