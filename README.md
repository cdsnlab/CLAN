CLAN: Self-supervised New Activity Detection in Sensor-based Smart Environments

With the rapid advancement of ubiquitous computing technology, human activity analysis based on time series data from a diverse range of sensors enables the delivery of more intelligent services. Despite the importance of exploring new activities in real-world scenarios, existing human activity recognition studies generally rely on predefined known activities and often overlook detecting new patterns (novelties) that have not been previously observed during training. Novelty detection in human activities becomes even more challenging due to (1) diversity of patterns within the same known activity, (2) shared patterns between known and new activities, and (3) differences in sensor properties of each activity dataset. We introduce CLAN, a two-tower model that leverages Contrastive Learning with diverse data Augmentation for New activity detection in sensor-based environments. CLAN simultaneously and explicitly utilizes multiple types of strongly shifted data as negative
samples in contrastive learning, effectively learning invariant representations that adapt to various pattern variations within the same activity. To enhance the ability to distinguish between known and new activities that share common features, CLAN incorporates both time and frequency domains, enabling the learning of multi-faceted discriminative representations. Additionally, we design an automatic selection mechanism of data augmentation methods tailored to each dataset’s properties, generating appropriate positive and negative pairs for contrastive learning. Comprehensive experiments on real-world datasets show that CLAN achieves a 9.24% improvement in AUROC compared to the best-performing baseline model.
