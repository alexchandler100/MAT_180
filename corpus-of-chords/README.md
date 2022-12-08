Team: Janice Adams, Keiran Hozven-Farley, Tori Tomlinson

***
# **Project: Corpus of Chords**
***
 * Data:  
    * Our project uses a dataset of [Bach Chorales](https://github.com/czhuang/JSB-Chorales-dataset) hosted on Github in MIDI note numbers.
    * This dataset translates all of the chords in the songs into 4 Dimensional vectors.
    * Under the folder Data\Corpi, we remove any repeated chords in all of the songs.   
    * Then Jsb_Vocab .  
      
 * Word2Vec:  
    * We used the skip gram model to find the probability of chords appearing next to each other from our dataset.
    * Generate the weights vector which  
      
 * K-cluster:  
    * Our K-cluster algorithm takes the weights from our Word2Vec function and creates clusters around the points.  
    * Then we find all the chords that are associated with each cluster. The similarity between chords are defined by the Euclidean distance between each point within a cluster.  
      
 * Table(Then talk about what the table represents. Music theory applications?) :  
        
    
## **How to use this machine learning project**  

   1. Data (Instructions state that user should be able to use their own data).  
      
   2. Import Python libraries (should we mention this?)  
   
   3. Run the Jupyter notebook (Does a user need to download the scripts or will cloning the repository be enough?)  
   


##
