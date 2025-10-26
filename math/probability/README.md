# Possion distribution 
Task 0

# "3. Initialize Exponential"

Modelon kohen deri sa ndondh nje ngjarje ne nje process Poisson.

Shembull intuitiv:
koha sa nje klienti i ri hyn ne dyqan
f(x;λ)=λe−λx,x≥0
𝜆 = rate (ngjarje për njësi kohe)

𝑥 = koha që kalon deri në ngjarje

Lidhja me Machine Learning
1.Simulime dhe modele probabilistike

Kur gjeneron kohë të rastësishme për ngjarje (queueing, failures, clicks, logins).

Shpesh përdoret në survival analysis dhe reliability engineering, të cilat mund të jenë pjesë e ML për predictive modeling.

2.Poisson Process & Exponential interarrival times

Në ML mund të modelosh ngjarje që ndodhin në kohë (events over time).

Shembull: Shuma e klikimeve për një faqe web; intervali mes klikimeve ndjek exponential distribution.

3.Neural Networks / Dropout Timing

Në simulime dhe stochastic processes mund të përdoret për të gjeneruar intervale të rastësishme për trajnim ose testim.

3.MLE (Maximum Likelihood Estimation)

Parametri 𝜆 mund të estimohen nga të dhë