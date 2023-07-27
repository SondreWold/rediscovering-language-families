# Rediscovering Language Family Taxonomies using Distributional Semantics
This project tries to see if static word embedding models trained on parallel Bible translations, using word2vec, is enough to cluster languages according to traditional language families. Models are trained on the same sections of the Bible (approximately 30k segmented lines), with the same hyperparameters, for a relatively short period of time. 

![image info](./img/early_attempt.png)
*Attempt at clustering some Indo-European and Afro-Asiatic langauges (with some obvious errors)*

## Data
I'm using the translations made available through the paper *A massively parallel corpus: the Bible in 100 languages*, by Christos Christodoulopoulos and Mark Steedman. The paper can be read [here](https://link.springer.com/article/10.1007/s10579-014-9287-y), and the data is available through the [NLPL](https://opus.nlpl.eu/bible-uedin.php). To run the scripts, download the languages you would like to work with through that url and place the extracted folder in the project root. For the languages that have been translated using MT, remove the *-MT* suffix on the XML. 