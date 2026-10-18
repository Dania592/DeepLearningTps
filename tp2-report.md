# Study report : TP2 - CNN and Transfer Learning : Age Estimation form Frace Images
## Part 1
MSE(Mean Squared Error) is in squared years, so hard to interpret so we use Mean Absolute Error (MAE) : 

Restored weights from epoch 19 (best val MAE = 11.68)
From-scratch CNN — Test MAE: 14.03 years
À titre de référence, un modèle trivial qui prédit toujours l’âge moyen du jeu d’entraînement (30,4 ans) obtient une MAE de test de 14,39 ans. 
Les courbes Train et Val sont presque superposées : il n’y a pas d’écart qui se creuse, donc pas d’overfitting. En revanche, l’erreur reste très élevée, autour de 11,7 ans. Le modèle n’arrive pas à bien apprendre, même sur les données d’entraînement : c’est de l’underfitting.
on remarque un pic important à l'epoque 18 : mais le point clé c'est que Train MAE et Val MAE font des pics en meme temps alors que le train loss mse descend de facon régulière, donc kes pics viennent de l'état du modèle, on peut aussi supposer que le learning rate est un peu élevé avec Adaman ce qui fait les sauter d'une époque à une autre don peut conclure que le moèdel n'a pas fini d'apprendre 

Sur-apprentissage ou sous-apprentissage ? Les courbes de MAE d’entraînement et de validation sont presque superposées tout au long de l’entraînement (écart inférieur à 0,5 an) : le modèle ne sur-apprend pas. En revanche, les deux erreurs restent élevées (≈ 11,5 à 12,5 ans) et plafonnent rapidement : le modèle est en sous-apprentissage. Avec environ 4 000 images d’entraînement et sans caractéristiques pré-apprises, un petit CNN entraîné from scratch ne parvient pas à extraire des caractéristiques faciales suffisamment discriminantes pour estimer l’âge.

On observe aussi des pics d’erreur aux époques 5, 13 et 18, qui apparaissent simultanément sur la MAE d’entraînement et de validation, alors que la loss d’entraînement continue de décroître régulièrement. Cela traduit une instabilité de l’entraînement (learning rate trop élevé pour Adam, statistiques de la BatchNorm en mode évaluation) plutôt qu’un sur-apprentissage.