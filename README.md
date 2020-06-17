## Healthcare Hackathon MRIQA 2020 - Intelligente Qualitätsanalyse von MRT Aufnahmen

#OPENKI #MRIQA

Willkommen auf der Projektseite zu unserem Hack im Bereich Quantenmedizin &amp; künstliche Intelligenz auf dem Healthcare Hackathon Mainz 2020. Auf dieser Seite findet Ihr alle notwendigen Informationen zu unserem Thema sowie Kontaktmöglichkeiten.

#### Vorstellung

Der Hack ist eine gemeinsame Initiative der Forschungsgruppe Medizin und Digitalisierung, dem Nationalen Neuroimaging Netzwerk sowie dem Geschäftsbereich IT und Medizintechnik am Universitätsklinikum Magdeburg. Die konkrete Ansprechpartner sind …

| Einrichtung | Otto-von-Guericke-Universität Magdeburg | Deutsches Zentrum für Neurodegenerative Erkrankungen | Universitätsklinikum Magdeburg |
| --- | --- | --- | --- |
| Bereich | [Forschungsgruppe Medizin und Digitalisierung](https://www.kneu.ovgu.de/MedDigit.html) | [Nationalen Neuroimaging Netzwerk](https://www.dzne.de/forschung/neuroimaging/) | [Geschäftsbereich IT und Medizintechnik](https://www.mrz.ovgu.de/) |
| Leitung | [Steffen Oeltze-Jafra](steffen.oeltze-jafra@med.ovgu.de)  | Emrah Düzel |  Robert Waschipky |
| Ansprechpartner im Hack | [Max Dünnwald](max.duennwald@med.ovgu.de) | [Laura Dobisch](laura.dobisch@dzne.de) | [Marko Rak](marko.rak@med.ovgu.de) |

#### Motivation

Sowohl bei der Patientenversorgung als auch bei der Durchführung medizinischer Studien spielen Magnetresonanztomographien (MRTs) eine zunehmend wichtigere Rolle. Im Betrieb eines größeren Klinikums oder im Rahmen einer größeren Studie werden tausende von Aufnahmen pro Jahr durchgeführt. Am Universitätsklinikum Magdeburg allein über 11.000 Untersuchungen im Jahr 2019. Ein Teil dieser Aufnahmen stellt sich wegen Artefakten bei der Bildgebung im Nachhinein als qualitativ unzureichend heraus und muss damit ggf. wiederholt werden. Aufgrund der begrenzten MRT-Zeit und hohen MRT-Kosten ist es das Ziel jedes Klinikums/jeder Studie die Anzahl an qualitative unzureichenden Aufnahmen durch Verbesserung des Prozessablaufes, des Personalschulung, der MRT-Parameter usw. zu senken. Als erster Schritt gilt es allerdings dieses Problem ersteinmal erkennen und quantifizieren zu können und dies möglichst vollautomatisch. Darum geht es uns bei diesem Hack.

#### Zielstellung

Entwerft in einem kleinen Team eine künstliche Intelligenz – basierend auf Convolutional Neural Networks (CNNs) – die verschieden Bildgebungsartefakte in MRT-Aufnahmen möglichst verlässlich erkennen und auch unterscheiden kann. Lasst eure künstliche Intelligenz gegen die der anderen Teams antreten. Es geht um Ruhm und Ehre und die beste Lösung wird prämiert…

#### Zeitplanung

Der Zeitplan des Hacks ist wiefolgt

| Datum | Uhrzeit | Inhalt |
| --- | --- | --- |
| 21.06. | 11:30 – 11:40 | Studioauftritt im GHH (Gutenberg Health Hub) |
| 21.06. | Ab 12:00 | Start der Team Coachings |
| 21.06. | Ab 12:00 | Start des Hacks |
| 22.06. | TBD | Abgabe der Lösungen |
| 22.06. | 12:00 – 13:00 | Prämierung der Teams |

Der Hack startet offiziell am 21.06. ab 12:00. Alle notwendigen Materialen stehen jedoch schon jetzt online zur Verfügung, sodass Ihr euch einen kleinen Vorsprung verschaffen könnt.

#### Voraussetzung

Grundsätzlich sind natürlich alle Hacker herzlich eingeladen teilzunehmen. Folgendes solltet Ihr jedoch mitbringen um richtig durchstarten zu können.

Expertise

- Erfahrungen in der Verarbeitung von Bilddaten
- Erfahrungen im Umgang mit medizinischen Bildern
- Expertise in Deep Learning und Convolutional Neural Networks
- Expertise in Anaconda, Python und PyTorch

Hardware

- Eine Grafikkarte mit viel Speicher
- Je mehr Hauptspeicher desto besser
- Eine verlässliche Kaffeemaschine …

#### Starterset

_Datensatz_

Als Datenbasis für den Hack dient uns der IXI-MRT-Datensatz. Bitte ladet euch unter dem nachstehenden Link die dort mit T1 und T2 gekennzeichneten Bilddaten herunter.

[https://brain-development.org/ixi-dataset/](https://brain-development.org/ixi-dataset/)

Wir nutzen nur einen Teil dieser beide Datensätze, deshalb entpackt bitte beide und zieht nur die Bilddaten des Hammersmith Hospital (diese tragen ein HH im Namen) heraus.

_Skripte_

Neben den Daten stellen wir euch auch ein paar Skripte zur Verfügung. Der IXI-Datensatz hat insgesamt eine hohe Datenqualität, Aufnahmeartefakte sind praktisch nicht vorhanden. Um dennoch eine artefaktbehaftete Datenbasis erzeugen zu können, haben wir euch ein Skript geschrieben

[To be linked]()

Dieses Skript liest die heruntergeladenen Bilddaten und generiert neue Bilddaten inkl. zufälligen Artefakten. Zudem generiert dieses Skript für jedes erzeugte Bild die Information ob Artefakte hinzugefügt wurden und welche Artefakte das jeweils waren.

Mit der nun passenden Datenbasis können wir auch schon starten. Um euch den Start zu erleichtern haben wir ein zweites Skript geschrieben, welches ein simples CNN auf den erzeugten Daten trainiert.

[To be linked]()

Eins vorweg, wir haben uns so garkeinen Kopf über die richtige Architektur und Parametrierung oder das Thema Data Augmentation gemacht, hier gibt es also noch deutliches Verbesserungspotential.

Ihr seid startbereit, meldet euch als Team bei [Marko Rak](marko.rak@med.ovgu.de) und dann kann es schon losgehen!

#### Prämierung

Es geht um die Wurst. Wer am Ende des Hacks die beste Lösung hat wird mit Ruhm und Ehre überschüttet Nein, Scherz beiseite, jede Lösung und jeder Lösungsweg bringt uns ein Stück weit voran.

Aber dennoch werden wir am Ende eine kleine Auswertung auf unseren eigenen echten Bilddaten machen und die beste Lösung/das beste Team prämieren 

#### Einreichung

Sobald eure Lösung einen Stand erreicht hat, den Ihr einreichen wollt, dann committed doch bitte hier im GitHub-Projekt eure Lösung im für euer Team vorgesehenen Bereich. Wenn es die Zeit zulässt machen wir vielleicht schon einen kleinen Testlauf von eure Lösung auf unseren Bilddaten und teilen euch mit wie gut Ihr seid, wer weiß das schon.

[To be linked]()

Euer Skript bzw. eure Skripte sollte möglichst genauso funktionieren wie das von uns zur Verfügung gestellt CNN-Skript CNN, damit wir keinen Aufwand bei der Anpassung auf unsere Daten haben.

#### Endergebnis

Wir werden sehen wer das Rennen macht…

| # | Teamname | Ergebnis | Kommentar |
| --- | --- | --- | --- |
| 01 | | | |
| 02 | | | |
| 03 | | | |
| ... | | | |
