# ANFIS
This repository contains an Type-1 Artificial Neural Fuzzy Inferece System in Python programming language.

At this moment this project only supports precedent fuzzy with the same number of labels. Besides, the learning algorithm implemented is the Hybrid Batch Online presented by Jang (1993). This method update the consequent labels using a Least Square Estimation and the precedent parameters with an Backpropagatino algorithm.

For extracting the features from the uttrances, I reccomend the code in the this
<a href=https://github.com/jameslyons/python_speech_features>repository</a>

### Dependencies
<ol>
	<li>Numpy 1.13.3</li>
    <li>SciPy 1.0.0</li>
</ol>

##### References
<ul>
	<li>
    <a href=http://ieeexplore.ieee.org/abstract/document/256541>Jang, J-SR. "ANFIS: adaptive-network-based fuzzy inference system." IEEE transactions on systems, man, and cybernetics 23.3 (1993): 665-685</a>
    </li>
    <li>
    <a href=https://www.amazon.com.br/Neuro-Fuzzy-Soft-Computing-Computational-Intelligence/dp/0132610663>Jang, Jyh-Shing Roger, Chuen-Tsai Sun, and Eiji Mizutani. "Neuro-fuzzy and soft computing: a computational approach to learning and machine intelligence." (1997)</a>
    </li>
</ul>

###### Note
<i>This project is in work, and may not be suitable for generic purposes. Right now this is being designed for Speech Recognition, more specifically for phoneme classification.</i>