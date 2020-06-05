from sklearn.feature_extraction.text import CountVectorizer

sentence1 = 'The Asian Summer Monsoon (ASM) is the largest meteorological pattern in the Northern Hemisphere (NH) summer season. Persistent convection and the large anticyclonic flow pattern in the upper troposphere and lower stratosphere (UTLS) associated with ASM produces a prominent enhancement of chemical species of pollution and biomass burning origins in the UTLS. The monsoon convection occurs over South, Southeast, and East Asia, a region of uniquely complex and rapidly changing emissions tied to both its high population density and significant economic growth. The coupling of the most polluted boundary layer on Earth to the largest dynamical system in the summer season through the deep monsoon convection has the potential to create significant chemical and climate impacts. An accurate representation of the ASM transport, chemical and microphysical processes in chemistry-climate models is much needed for characterizing ASM chemistry-climate interactions and for predicting its future impact in a changing climate.'
sentence2 = 'The Clouds, Aerosol and Monsoon Processes-Philippines Experiment (CAMP²Ex) is a NASA airborne mission with the goal to characterize the role of anthropogenic and natural aerosol particles in modulating the frequency and amount of warm and mixed phase precipitation in the vicinity of the Philippines during the Southwest Monsoon. In partnership with Philippine research and operational weather communities, CAMP²Ex provides a comprehensive 4-D observational view of the environment of the Philippines and its neighboring waters in terms of microphysical, hydrological, dynamical, thermodynamical and radiative properties of the environment, targeting the environment of shallow cumulus and cumulus congestus clouds. . The NASA P-3 conducted nineteen research flights, during which NCAR RD41 dropsondes were released.'
sentences = [sentence1 , sentence2]

vectorizer = CountVectorizer(min_df=0, lowercase=False)
vectorizer.fit(sentences)
print1 = vectorizer.vocabulary_

print(print1)

print2 = vectorizer.transform(sentences).toarray()
print(print2)
