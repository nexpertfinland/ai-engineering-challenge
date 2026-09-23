# AI Finland workshop: toteutus ja tulokset

## Mitä tehtiin

Tavoite on vastata kysymyksiin paikallisen Wikipedia-aineiston perusteella.
Ratkaisu säilyttää alkuperäisen `llama3.2:3b`-mallin ja toimii Ollamalla ilman
maksullista API-avainta. Muutokset kohdistuvat tiedonhakuun, vastauksen
muodostamiseen ja tuloksen tarkistamiseen.

1. **Kappaleet säilytetään.** Alkuperäinen 500 merkin leikkaus saattoi katkaista
   vastauksen tai sen selittävän lauseen. Nyt tavallinen kappale pysyy kokonaisena.
   Yli 220 sanan kappaleet jaetaan läheltä lauserajaa, 40 sanan limityksellä.
2. **Haku painottaa erottelevia sanoja.** BM25 vähentää yleisten sanojen ja
   toistojen ylivaltaa. TF-IDF täydentää hakua yhden ja kahden sanan ilmauksilla.
   Yhdistelmän painot ovat BM25 0,6 ja TF-IDF 0,4. Myös artikkelin otsikko on mukana.
3. **Mallille annetaan kolme katkelmaa.** Lähes päällekkäiset katkelmat suodatetaan.
   Tämä parantaa vastauksen löytymistä mutta kasvattaa syötteen tokenmäärää.
4. **Vastaus on lyhyt ja perusteltu.** Mallilta pyydetään vastausfraasi ja sitä
   tukevan katkelman numero JSON-muodossa. Generoinnin temperature on 0 ja
   enimmäispituus 128 tokenia.
5. **Lähde tarkistetaan.** Vastauksen pitää löytyä valitusta katkelmasta
   kirjainkoosta ja välilyöntien määrästä riippumatta. Virheellinen rakenne,
   keksitty lähdenumero tai katkelmaan perustumaton vastaus hylätään.
   Lähteeksi ei merkitä automaattisesti kaikkia haettuja artikkeleita.
6. **Vertailu on toistettavissa.** Alkuperäinen haku ja prompti ovat
   `rag/baseline.py`-tiedostossa. Harva matriisi säästää muistia muuttamatta
   lähtöratkaisun järjestystä. Vastaavuus alkuperäiseen notebookiin tarkistettiin
   kaikilla 40 kehityskysymyksellä.

## Mitattu tiedonhaku

Mittaus 23.9.2026, tehtävän alkuperäinen 40 kysymyksen kehitysjoukko.
"Vastaus mukana" edellyttää sekä oikeaa artikkelia että jonkin hyväksytyn
vastauksen esiintymistä haetussa katkelmassa. Se ei vielä tarkoita, että malli
tuottaa oikean vastauksen.

| Haku | Katkelmia | Oikea artikkeli mukana | Vastaus mukana | Kontekstin merkit keskimäärin |
|---|---:|---:|---:|---:|
| Alkuperäinen | 1 | 12/40 (30 %) | 7/40 (17,5 %) | 500 |
| Alkuperäinen, vertailu samalla katkelmamäärällä | 3 | 17/40 (42,5 %) | 10/40 (25 %) | 1 500 |
| Parannettu | 1 | 38/40 (95 %) | 33/40 (82,5 %) | 761 |
| Parannettu, käytössä oleva asetus | 3 | 39/40 (97,5 %) | 38/40 (95 %) | 2 268 |

Kehitysjoukolla kokeiltiin lisäksi pelkkää TF-IDF:ää ja pelkkää BM25:tä.
Kaikki kolme löysivät vastauksen 38/40 kysymykseen kolmella katkelmalla.
Painotettua yhdistelmää ei valittu piilotetun testijoukon vastausten perusteella.

## Varsinaiset malliajot

Kehitysajo valmistui 23.9.2026. Molemmat ratkaisut käyttivät samaa
`llama3.2:3b`-mallia (mallitunniste `a80c4f17acd5`), Ollamaa 0.34.3 ja
Pythonia 3.11.16 GitHubin Ubuntu-ajoympäristöissä. Malli ajettiin paikallisesti
näissä ympäristöissä, ilman maksullista kielimallirajapintaa.

| Mittari, 40 kehityskysymystä | Alkuperäinen | Parannettu |
|---|---:|---:|
| Kehityspisteet / 100 | 15,97 | 76,98 |
| Vastausten F1 | 0,1246 | 0,7373 |
| Täsmälleen oikeat vastaukset | 7,5 % | 65 % |
| Oikea lähdeviite | 30 % | 90 % |
| Tokenit yhteensä | 7 969 | 26 738 |
| Syötetokenit | 7 060 | 26 110 |
| Vastaustokenit | 909 | 628 |
| Ajoaika | 133,42 s | 488,91 s |

Kehityspisteet paranivat 61,01 pistettä. Tokenkulutus kasvoi noin 3,4-kertaiseksi:
lyhyemmät vastaukset säästävät vastaustokeneita, mutta pidempi lähdekonteksti
kasvattaa syötetokeneita enemmän. Tämä toteutus parantaa ensisijaisesti laatua.
Ajoajat sisältävät mallikutsut ja riippuvat myös ajokoneesta ja mallin latauksesta.

Kehitysajo ja ladattavat JSON-tulokset:
https://github.com/nexpertfinland/ai-engineering-challenge/actions/runs/35865778833

### Lopputesti: 80 erillistä kysymystä

Molemmat ajot valmistuivat onnistuneesti. Testattu ratkaisun versio:
`8c72d6931b0035ec3122c10e96f01342106cfa36`. Myöhemmät muutokset lisäävät raportin
ja käynnistysohjeet; pisteytettyä hakua tai vastauslogiikkaa ei muutettu.

| Mittari, 80 testikysymystä | Alkuperäinen | Parannettu |
|---|---:|---:|
| **Tehtävän pisteet / 100** | **16,01** | **71,71** |
| Vastausten F1 | 0,1126 | 0,6870 |
| Täsmälleen oikeat vastaukset | 6/80 (7,5 %) | 49/80 (61,25 %) |
| Oikea lähdeviite | 28/80 (35 %) | 67/80 (83,75 %) |
| Suoritusvirheet | 0 | 0 |
| Tokenit yhteensä | 16 127 | 51 351 |
| Syötetokenit | 14 203 | 49 981 |
| Vastaustokenit | 1 924 | 1 370 |
| Ajoaika | 358,8 s (noin 6 min) | 1 415,2 s (noin 24 min) |
| Keskimääräinen vastausaika | 4,49 s | 17,69 s |

**Laatupisteet paranivat 55,70 pistettä.** Parannus näkyi myös kysymyksissä,
joita ei käytetty asetusten valintaan. Tokenkulutus nousi noin 3,2-kertaiseksi
ja ajoaika oli tässä ympäristössä noin nelinkertainen. Tämä ratkaisu menestyy
laadun vertailussa; alkuperäinen ratkaisu kuluttaa vähemmän tokeneita.

Lopputestin tulos, lokit ja ladattavat JSON-tiedostot:
https://github.com/nexpertfinland/ai-engineering-challenge/actions/runs/35866150659

Mittaukset ilman yksittäisiä aineiston kysymyksiä:
[reports/benchmark-results.json](reports/benchmark-results.json).

Kaikki 15 yksikkötestiä läpäistiin paikallisesti ja jokaisessa neljässä
GitHub-arviointityössä. Notebookin rakenne, kaikki koodisolujen syntaksit sekä
aloitus- ja hakusolujen suoritus tarkistettiin. Varsinaiset mallimittaukset ajettiin
komentoriviltä samalla `RAGPipeline.answer_question`-funktiolla, jota notebook
käyttää. Notebookiin ei ole lisätty jälkikäteen tekaistuja solutulosteita.

## Ajaminen omalla koneella

Ollaman pitää olla käynnissä ja mallin ladattu:

```powershell
ollama pull llama3.2:3b
uv sync
uv run python -m jupyterlab notebooks/challenge.ipynb
```

Notebook: **Run → Run All Cells**. Käynnissä olevan palvelimen PowerShell-ikkuna
jätetään auki. Koko notebook tekee yhden esimerkkivastauksen, 40 kehityskysymystä
ja 80 testikysymystä. Kesto riippuu koneesta.

Pelkkä tiedonhaun vertailu, ilman mallikutsuja:

```powershell
uv run python scripts/benchmark.py --split retrieval
```

Molempien ratkaisujen kehitysajo:

```powershell
uv run python scripts/benchmark.py --split dev --pipeline both
```

Molempien ratkaisujen lopullinen arviointi alkuperäisellä arviointikoodilla:

```powershell
uv run python scripts/benchmark.py --split hidden --pipeline both
```

JSON-tulokset tallentuvat `results/`-kansioon. Kehitysajon yksittäiset vastaukset
auttavat virheiden tutkimisessa; niitä tai aineistoa ei commitoida repoon.
Piilotetusta testistä tallennetaan vain alkuperäisen arviointikoodin yhteenveto.

Testit:

```powershell
uv run python -m unittest discover -s tests -v
```

Saman vertailun voi käynnistää GitHubissa: **Actions → Workshop benchmark → Run
workflow**. Valitse `dev` kehitysajoon tai `hidden` lopulliseen arviointiin.
Kumpikin vaihtoehto ajaa alkuperäisen ja parannetun ratkaisun erillisissä
Ubuntu-ympäristöissä samalla mallilla. Ajoon kuuluvat myös yksikkötestit ja
arviointikoodin tarkistussummien tarkistus. Tulokset löytyvät ajon Artifacts-kohdasta.
Työnkulku käynnistyy vain pyydettäessä.

## Rajaukset ja reilu vertailu

- `eval/harness.py`, `eval/metrics.py` ja `rag/dataset.py` ovat alkuperäisiä.
  Aineiston jakoperusteita, kysymyksiä tai kultavastauksia ei muutettu.
- Ratkaisu käyttää vastaamiseen vain korpusta ja kysymystä. Se ei lue
  `dev_qa.json`- tai `test_qa.json`-tiedostoja eikä sisällä kysymyskohtaisia vastauksia.
- Kaikki generointikutsut käyttävät `call_llm()`-funktiota. Myös hylätyt vastaukset
  kuluttavat tokeneita, jotka lasketaan mukaan.
- Mallin antama lyhyt vastaus voi esiintyä lähteessä ja silti vastata väärään asiaan.
  Lähdetarkistus vähentää perusteettomia vastauksia, mutta ei takaa oikeellisuutta.
- Kolme katkelmaa kuluttaa enemmän syötetokeneita kuin yksi. Ratkaisu tavoittelee
  parempaa laatua; pienintä mahdollista tokenkulutusta ei luvata.
- Pieni kehitysjoukko ei takaa samanlaista parannusta kaikissa aineistoissa.
- Raportti kertoo yhden ajokerran tulokset. Alkuperäinen ratkaisu käyttää mallin
  alkuperäisiä generointioletuksia, joten sen vastaukset voivat vaihdella ajosta
  toiseen. Myös eri laitteisto tai malliversio voi muuttaa tulosta ja ajoaikaa.
- Työympäristön hakutesti käytti SQuADin tekijöiden alkuperäistä validation-JSONia
  ja muuttamatonta `build_splits()`-funktiota. GitHubin malliajot käyttävät repon
  alkuperäistä Hugging Face -latausta. Korpuksen SHA-256 kirjataan tuloksiin
  aineiston vastaavuuden tarkistamiseksi. Paikallisen hakutestin ja GitHub-ajon
  korpuksen SHA-256 täsmäsi: `84acadde3996684ab48a764136af52b3e961752d01554af8287724860a2c9014`.

## Lähteet

- Tehtävän alkuperä: https://github.com/BruvoAI/ai-engineering-challenge
- SQuAD 1.1, Rajpurkar ym., CC BY-SA 4.0: https://rajpurkar.github.io/SQuAD-explorer/
- Aineiston tekijöiden repo: https://github.com/rajpurkar/SQuAD-explorer
- TF-IDF: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- Ollaman viesti-, JSON- ja tokenrajapinta: https://docs.ollama.com/api/chat
