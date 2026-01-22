#!/usr/bin/env python3
"""
100k STT Dataset Generator for Travel Order Resolver.

Generates a large-scale dataset simulating realistic Whisper
speech-to-text transcription output for French railway travel requests.

Dataset schema:
- sentence_id: Unique identifier (STT000001 format)
- sentence: Input text (may contain STT errors)
- intent: TRIP | NOT_TRIP | UNKNOWN
- language: FRENCH | ENGLISH | SPANISH | GERMAN | ITALIAN | UNKNOWN
- departure: Clean departure station name
- destination: Clean destination station name
- intermediate: Clean intermediate station name (for "via" routes)

Usage:
    python generate_stt_dataset.py [--output DIR] [--count N] [--seed S]

Example:
    python generate_stt_dataset.py --count 100000 --seed 42 --output datasets/generated/
"""

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from data import StationDatabase  # noqa: E402
from data.stt_augmentation import STTAugmenter, create_augmenter  # noqa: E402

# =============================================================================
# DATASET SCHEMA
# =============================================================================

IntentType = Literal["TRIP", "NOT_TRIP", "UNKNOWN"]
LanguageType = Literal["FRENCH", "ENGLISH", "SPANISH", "GERMAN", "ITALIAN", "UNKNOWN"]


@dataclass
class DatasetEntry:
    """Single entry in the STT dataset."""

    sentence_id: str
    sentence: str
    intent: IntentType
    language: LanguageType
    departure: str = ""
    destination: str = ""
    intermediate: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for CSV/JSON export."""
        return {
            "sentence_id": self.sentence_id,
            "sentence": self.sentence,
            "intent": self.intent,
            "language": self.language,
            "departure": self.departure,
            "destination": self.destination,
            "intermediate": self.intermediate,
        }


# =============================================================================
# FRENCH TRIP TEMPLATES (200+)
# =============================================================================

# Basic patterns (from original + expanded)
TRIP_TEMPLATES_FR_BASIC = [
    "Je voudrais aller de {dep} a {dest}",
    "Je veux aller de {dep} a {dest}",
    "Je souhaite aller de {dep} a {dest}",
    "Je dois aller de {dep} a {dest}",
    "Je vais de {dep} a {dest}",
    "Aller de {dep} a {dest}",
    "De {dep} a {dest}",
    "{dep} {dest}",
    "{dep} - {dest}",
    "{dep} vers {dest}",
    "{dep} direction {dest}",
    "Je voudrais prendre un train de {dep} a {dest}",
    "Je veux prendre le train de {dep} a {dest}",
    "Prendre le train de {dep} a {dest}",
    "Un train de {dep} a {dest}",
    "Train de {dep} a {dest}",
    "Un billet de {dep} a {dest}",
    "Billet {dep} {dest}",
    "Je pars de {dep} pour aller a {dest}",
    "Je pars de {dep} et je vais a {dest}",
    "Depart de {dep} arrivee a {dest}",
    "Depart {dep} arrivee {dest}",
    "Partir de {dep} pour {dest}",
    "Partir de {dep} vers {dest}",
    "Je voudrais me rendre a {dest} depuis {dep}",
    "Je veux me rendre a {dest} en partant de {dep}",
    "Me rendre de {dep} a {dest}",
    "Je me rends a {dest} depuis {dep}",
    "Je voyage de {dep} a {dest}",
    "Voyage de {dep} a {dest}",
    "Voyager de {dep} vers {dest}",
    "Comment aller de {dep} a {dest}",
    "Comment aller de {dep} a {dest} ?",
    "Quel train pour aller de {dep} a {dest}",
    "Quel train de {dep} a {dest} ?",
    "Y a-t-il un train de {dep} a {dest}",
    "Y a-t-il un train de {dep} a {dest} ?",
    "Est-ce qu'il y a un train de {dep} a {dest}",
    "Je vais a {dest} depuis {dep}",
    "{dep} puis {dest}",
    "De {dep} jusqu'a {dest}",
    "Depuis {dep} jusqu'a {dest}",
    "Direction {dest} au depart de {dep}",
]

# Informal/Conversational patterns (50+)
TRIP_TEMPLATES_FR_INFORMAL = [
    "Ben je veux aller de {dep} a {dest}",
    "Bon alors de {dep} a {dest}",
    "Genre {dep} {dest} quoi",
    "En gros {dep} a {dest}",
    "Faut que j'aille de {dep} a {dest}",
    "Faut y aller de {dep} a {dest}",
    "J'ai besoin d'aller de {dep} a {dest}",
    "La je voudrais partir de {dep} pour {dest}",
    "Voila je veux aller de {dep} a {dest}",
    "Bah de {dep} a {dest}",
    "Ouais donc {dep} a {dest}",
    "Moi je vais de {dep} a {dest}",
    "Tu vois je veux aller de {dep} a {dest}",
    "En fait je vais de {dep} a {dest}",
    "Donc voila {dep} {dest}",
    "C'est bon {dep} a {dest}",
    "Ok donc {dep} vers {dest}",
    "Allez {dep} {dest}",
    "Hop {dep} a {dest}",
    "Vite un train de {dep} a {dest}",
    "Je file de {dep} a {dest}",
    "Je trace de {dep} a {dest}",
    "Je me casse de {dep} pour {dest}",
    "Moi c'est {dep} {dest}",
    "Pour moi c'est {dep} a {dest}",
    "Je dois me rendre de {dep} a {dest}",
    "Faudrait que je parte de {dep} pour {dest}",
    "Il me faut un train de {dep} a {dest}",
    "Y'a moyen d'aller de {dep} a {dest}",
    "C'est possible d'aller de {dep} a {dest}",
    "Je peux avoir un billet de {dep} a {dest}",
    "Un p'tit train de {dep} a {dest}",
    "Juste {dep} {dest}",
    "Simple {dep} a {dest}",
    "Tranquille {dep} vers {dest}",
    "Vite fait {dep} {dest}",
    "Rapide {dep} a {dest}",
    "Direct {dep} {dest}",
    "Allez hop de {dep} a {dest}",
    "Bon ben {dep} {dest} alors",
]

# With hesitations built into template (40+)
TRIP_TEMPLATES_FR_HESITATION = [
    "Euh je voudrais aller de {dep} a {dest}",
    "Hmm de {dep} a {dest}",
    "Alors euh {dep} a {dest}",
    "Ben euh {dep} vers {dest}",
    "Donc euh de {dep} a {dest}",
    "Euh un train de {dep} a {dest}",
    "Je veux euh aller de {dep} a {dest}",
    "Euh billet {dep} {dest}",
    "Hmm voyons {dep} a {dest}",
    "Attends euh {dep} {dest}",
    "Euh bon {dep} a {dest}",
    "Donc voila euh de {dep} a {dest}",
    "Comment dire euh {dep} vers {dest}",
    "C'est euh de {dep} a {dest}",
    "Bon euh je pars de {dep} pour {dest}",
    "Je voudrais euh enfin de {dep} a {dest}",
    "Alors alors {dep} a {dest}",
    "Hmm hmm {dep} vers {dest}",
    "Bon bon {dep} {dest}",
    "Voila voila de {dep} a {dest}",
    "Euh en fait de {dep} a {dest}",
    "Bah euh {dep} a {dest} quoi",
    "Donc donc {dep} vers {dest}",
    "Je je voudrais de {dep} a {dest}",
    "Un un train de {dep} a {dest}",
    "Euh attendez {dep} a {dest}",
    "Comment ca s'appelle euh {dep}",
    "Oui oui {dep} a {dest}",
    "Ah oui {dep} {dest}",
    "Ah {dep} vers {dest}",
]

# Regional/Colloquial (30+)
TRIP_TEMPLATES_FR_REGIONAL = [
    "Ch'veux aller de {dep} a {dest}",
    "J'veux y aller de {dep} a {dest}",
    "Faut qu'j'aille de {dep} a {dest}",
    "J'dois partir de {dep} pour {dest}",
    "M'faut un train de {dep} a {dest}",
    "J'aimerais bien aller de {dep} a {dest}",
    "Ca s'rait bien d'aller de {dep} a {dest}",
    "On va de {dep} a {dest}",
    "Nous on part de {dep} pour {dest}",
    "Moi j'vais de {dep} a {dest}",
    "Putain faut y aller de {dep} a {dest}",
    "Bordel je dois aller de {dep} a {dest}",
    "Allez on y va de {dep} a {dest}",
    "Vas-y {dep} a {dest}",
    "Steuplait {dep} {dest}",
    "Siouplait de {dep} a {dest}",
    "Dis donc {dep} a {dest}",
    "Dis {dep} {dest}",
    "Tiens {dep} vers {dest}",
    "Bref {dep} a {dest}",
    "Bon bref de {dep} a {dest}",
    "Enfin bref {dep} {dest}",
    "Quoi qu'il en soit {dep} a {dest}",
    "Du coup {dep} vers {dest}",
    "Du coup je vais de {dep} a {dest}",
    "Genre je pars de {dep} pour aller a {dest}",
    "Style {dep} {dest}",
    "Wesh {dep} a {dest}",
    "Yo {dep} {dest}",
    "Hep {dep} vers {dest}",
]

# With context (work, meeting, etc.) (30+)
TRIP_TEMPLATES_FR_CONTEXT = [
    "Pour le travail je dois aller de {dep} a {dest}",
    "Pour le boulot de {dep} a {dest}",
    "Mon rdv est a {dest} je pars de {dep}",
    "J'ai une reunion a {dest} depuis {dep}",
    "Je dois rejoindre quelqu'un a {dest} de {dep}",
    "Vacances de {dep} a {dest}",
    "Week-end a {dest} depuis {dep}",
    "Visite a {dest} je pars de {dep}",
    "Je vais voir ma famille a {dest} de {dep}",
    "Retour a {dest} depuis {dep}",
    "Aller a {dest} pour les vacances de {dep}",
    "Business trip de {dep} a {dest}",
    "Deplacement pro {dep} {dest}",
    "Formation a {dest} de {dep}",
    "Entretien a {dest} depuis {dep}",
    "Rdv medical a {dest} de {dep}",
    "Mariage a {dest} je pars de {dep}",
    "Enterrement a {dest} depuis {dep}",
    "Urgence a {dest} de {dep}",
    "Urgent de {dep} a {dest}",
    "Vite a {dest} depuis {dep}",
    "Le plus tot possible de {dep} a {dest}",
    "Demain de {dep} a {dest}",
    "Ce soir {dep} a {dest}",
    "Maintenant de {dep} a {dest}",
    "Aujourd'hui {dep} vers {dest}",
    "La semaine prochaine de {dep} a {dest}",
    "Lundi de {dep} a {dest}",
    "Ce week-end {dep} {dest}",
    "Pendant les fetes de {dep} a {dest}",
]

# Questions varied (20+)
TRIP_TEMPLATES_FR_QUESTIONS = [
    "C'est possible {dep} {dest}",
    "Vous avez quoi entre {dep} et {dest}",
    "Il y a des trains de {dep} a {dest}",
    "Quand part le prochain train de {dep} a {dest}",
    "A quelle heure le train de {dep} a {dest}",
    "Combien coute le trajet de {dep} a {dest}",
    "Ca coute combien {dep} {dest}",
    "Le prix pour {dep} {dest}",
    "Duree du trajet {dep} a {dest}",
    "Combien de temps de {dep} a {dest}",
    "Il faut combien de temps de {dep} a {dest}",
    "C'est loin {dep} a {dest}",
    "C'est direct de {dep} a {dest}",
    "Il y a des correspondances de {dep} a {dest}",
    "Le meilleur trajet de {dep} a {dest}",
    "Le plus rapide de {dep} a {dest}",
    "Le moins cher de {dep} a {dest}",
    "Quel TGV de {dep} a {dest}",
    "Quel TER de {dep} a {dest}",
    "Quelle ligne de {dep} a {dest}",
    "Comment y aller de {dep} a {dest}",
    "Par ou passer de {dep} a {dest}",
]

# Destination only (30+)
TRIP_TEMPLATES_FR_DEST_ONLY = [
    "Je vais a {dest}",
    "Je voudrais aller a {dest}",
    "Direction {dest}",
    "Destination {dest}",
    "Faut que j'aille a {dest}",
    "Je dois me rendre a {dest}",
    "Un train pour {dest}",
    "Un billet pour {dest}",
    "Pour {dest} s'il vous plait",
    "Vers {dest}",
    "{dest} s'il vous plait",
    "Aller a {dest}",
    "Je pars a {dest}",
    "Je file a {dest}",
    "Je vais jusqu'a {dest}",
    "Jusqu'a {dest}",
    "En direction de {dest}",
    "A destination de {dest}",
    "Cap sur {dest}",
    "On va a {dest}",
    "Moi je vais a {dest}",
    "Direct {dest}",
    "TGV pour {dest}",
    "TER pour {dest}",
    "Intercites pour {dest}",
    "Le prochain pour {dest}",
    "N'importe quel train pour {dest}",
    "Peu importe je veux aller a {dest}",
    "Emmene moi a {dest}",
    "Je veux rejoindre {dest}",
]

# Polite/Formal (20+)
TRIP_TEMPLATES_FR_POLITE = [
    "Bonjour je voudrais aller de {dep} a {dest}",
    "Bonjour un billet de {dep} a {dest} s'il vous plait",
    "S'il vous plait un trajet de {dep} a {dest}",
    "Pourriez-vous me trouver un train de {dep} a {dest}",
    "J'aimerais aller de {dep} a {dest}",
    "J'aimerais un billet de {dep} a {dest}",
    "Serait-il possible d'avoir un train de {dep} a {dest}",
    "Auriez-vous un train de {dep} a {dest}",
    "Je souhaiterais me rendre de {dep} a {dest}",
    "Merci de me trouver un trajet de {dep} a {dest}",
    "Je vous remercie de {dep} a {dest}",
    "Excusez-moi de {dep} a {dest}",
    "Pardon de {dep} a {dest} s'il vous plait",
    "Je voudrais reserver de {dep} a {dest}",
    "Une reservation de {dep} a {dest}",
    "Bonsoir de {dep} a {dest}",
    "Bonne journee et de {dep} a {dest}",
    "Cordialement de {dep} a {dest}",
    "Si possible de {dep} a {dest}",
    "Dans la mesure du possible {dep} {dest}",
]

# Combine all French TRIP templates
TRIP_TEMPLATES_FR = (
    TRIP_TEMPLATES_FR_BASIC
    + TRIP_TEMPLATES_FR_INFORMAL
    + TRIP_TEMPLATES_FR_HESITATION
    + TRIP_TEMPLATES_FR_REGIONAL
    + TRIP_TEMPLATES_FR_CONTEXT
    + TRIP_TEMPLATES_FR_QUESTIONS
    + TRIP_TEMPLATES_FR_DEST_ONLY
    + TRIP_TEMPLATES_FR_POLITE
)

# =============================================================================
# FRENCH INTERMEDIATE TEMPLATES (30+)
# =============================================================================

TRIP_INTERMEDIATE_TEMPLATES_FR = [
    # Basic patterns
    "Je voudrais aller de {dep} a {dest} en passant par {via}",
    "De {dep} a {dest} via {via}",
    "De {dep} a {dest} en passant par {via}",
    "{dep} {dest} via {via}",
    "Aller de {dep} a {dest} avec un arret a {via}",
    "Je veux passer par {via} pour aller de {dep} a {dest}",
    "Train de {dep} a {dest} passant par {via}",
    "{dep} puis {via} puis {dest}",
    "De {dep} vers {dest} avec correspondance a {via}",
    "Je pars de {dep}, je passe par {via} et j'arrive a {dest}",
    # Informal/Conversational
    "Euh de {dep} a {dest} en passant par {via}",
    "Ben {dep} puis {via} puis {dest}",
    "Faut que j'aille de {dep} a {dest} via {via}",
    "Genre {dep} {via} {dest}",
    "En gros {dep} a {dest} via {via}",
    "Je veux faire {dep} {via} {dest}",
    "Trajet {dep} {via} {dest}",
    "Billet {dep} {dest} via {via}",
    # With hesitations
    "Euh alors de {dep} a {dest} euh en passant par {via}",
    "Hmm {dep} puis euh {via} puis {dest}",
    "Je voudrais euh de {dep} a {dest} via {via}",
    # Questions
    "Y a-t-il un train de {dep} a {dest} via {via}",
    "C'est possible de passer par {via} pour aller de {dep} a {dest}",
    "Comment aller de {dep} a {dest} en passant par {via}",
    # Polite
    "Bonjour je voudrais aller de {dep} a {dest} en passant par {via}",
    "S'il vous plait de {dep} a {dest} via {via}",
    "J'aimerais passer par {via} pour aller de {dep} a {dest}",
    # Context
    "Demain de {dep} a {dest} via {via}",
    "Pour le travail {dep} a {dest} en passant par {via}",
    "Urgent de {dep} a {dest} via {via}",
]

# =============================================================================
# ENGLISH TRIP TEMPLATES (30+)
# =============================================================================

TRIP_TEMPLATES_EN = [
    "I want to go from {dep} to {dest}",
    "I would like to go from {dep} to {dest}",
    "I need to travel from {dep} to {dest}",
    "Train from {dep} to {dest}",
    "A ticket from {dep} to {dest}",
    "Ticket from {dep} to {dest} please",
    "{dep} to {dest}",
    "{dep} to {dest} please",
    "From {dep} to {dest}",
    "I'm going from {dep} to {dest}",
    "How do I get from {dep} to {dest}",
    "What's the best way from {dep} to {dest}",
    "Is there a train from {dep} to {dest}",
    "Can I get a train from {dep} to {dest}",
    "I need a ticket from {dep} to {dest}",
    "One way from {dep} to {dest}",
    "Round trip from {dep} to {dest}",
    "Departing from {dep} arriving at {dest}",
    "Departure {dep} arrival {dest}",
    "Going to {dest} from {dep}",
    "Traveling from {dep} to {dest}",
    "Journey from {dep} to {dest}",
    "Route from {dep} to {dest}",
    "Getting from {dep} to {dest}",
    "Take me from {dep} to {dest}",
    "I want a train from {dep} to {dest}",
    "Book a ticket from {dep} to {dest}",
    "Reserve from {dep} to {dest}",
    "Hi I need to go from {dep} to {dest}",
    "Hello train from {dep} to {dest}",
]

# English intermediate templates
TRIP_INTERMEDIATE_TEMPLATES_EN = [
    "From {dep} to {dest} via {via}",
    "I want to go from {dep} to {dest} through {via}",
    "{dep} to {dest} with a stop at {via}",
    "Train from {dep} to {dest} via {via}",
    "I need to go from {dep} to {dest} passing through {via}",
    "{dep} then {via} then {dest}",
    "Can I get a train from {dep} to {dest} via {via}",
    "Ticket from {dep} to {dest} stopping at {via}",
    "From {dep} via {via} to {dest}",
    "Journey from {dep} to {dest} with connection at {via}",
]

# =============================================================================
# SPANISH TRIP TEMPLATES (15+)
# =============================================================================

TRIP_TEMPLATES_ES = [
    "Quiero ir de {dep} a {dest}",
    "Necesito ir de {dep} a {dest}",
    "Un billete de {dep} a {dest}",
    "De {dep} a {dest} por favor",
    "Tren de {dep} a {dest}",
    "Como llego de {dep} a {dest}",
    "Hay trenes de {dep} a {dest}",
    "Viaje de {dep} a {dest}",
    "Voy de {dep} a {dest}",
    "Salida {dep} llegada {dest}",
    "Me gustaria ir de {dep} a {dest}",
    "Por favor de {dep} a {dest}",
    "Un pasaje de {dep} a {dest}",
    "Quisiera viajar de {dep} a {dest}",
    "Reservar de {dep} a {dest}",
]

# Spanish intermediate templates
TRIP_INTERMEDIATE_TEMPLATES_ES = [
    "De {dep} a {dest} pasando por {via}",
    "Quiero ir de {dep} a {dest} via {via}",
    "{dep} a {dest} con parada en {via}",
    "Tren de {dep} a {dest} via {via}",
    "{dep} luego {via} luego {dest}",
]

# =============================================================================
# GERMAN TRIP TEMPLATES (15+)
# =============================================================================

TRIP_TEMPLATES_DE = [
    "Ich mochte von {dep} nach {dest} fahren",
    "Von {dep} nach {dest} bitte",
    "Eine Fahrkarte von {dep} nach {dest}",
    "Zug von {dep} nach {dest}",
    "Wie komme ich von {dep} nach {dest}",
    "Gibt es einen Zug von {dep} nach {dest}",
    "Ich brauche eine Verbindung von {dep} nach {dest}",
    "Reise von {dep} nach {dest}",
    "Fahrt von {dep} nach {dest}",
    "Ich fahre von {dep} nach {dest}",
    "Abfahrt {dep} Ankunft {dest}",
    "Bitte von {dep} nach {dest}",
    "Ein Ticket von {dep} nach {dest}",
    "Verbindung {dep} {dest}",
    "Nach {dest} von {dep}",
]

# German intermediate templates
TRIP_INTERMEDIATE_TEMPLATES_DE = [
    "Von {dep} nach {dest} uber {via}",
    "Ich mochte von {dep} nach {dest} uber {via} fahren",
    "{dep} nach {dest} mit Halt in {via}",
    "Zug von {dep} nach {dest} uber {via}",
    "{dep} dann {via} dann {dest}",
]

# =============================================================================
# ITALIAN TRIP TEMPLATES (10+)
# =============================================================================

TRIP_TEMPLATES_IT = [
    "Vorrei andare da {dep} a {dest}",
    "Un biglietto da {dep} a {dest}",
    "Treno da {dep} a {dest}",
    "Come arrivo da {dep} a {dest}",
    "Da {dep} a {dest} per favore",
    "Viaggio da {dep} a {dest}",
    "Partenza {dep} arrivo {dest}",
    "Vado da {dep} a {dest}",
    "Mi serve un treno da {dep} a {dest}",
    "Per favore da {dep} a {dest}",
]

# Italian intermediate templates
TRIP_INTERMEDIATE_TEMPLATES_IT = [
    "Da {dep} a {dest} passando per {via}",
    "Vorrei andare da {dep} a {dest} via {via}",
    "{dep} a {dest} con fermata a {via}",
    "Treno da {dep} a {dest} via {via}",
    "{dep} poi {via} poi {dest}",
]

# =============================================================================
# MIXED/CODE-SWITCHING TEMPLATES (10+)
# =============================================================================

TRIP_TEMPLATES_MIXED = [
    "I want aller a {dest}",
    "Je want to go to {dest}",
    "I need un train de {dep} a {dest}",
    "Un ticket from {dep} to {dest}",
    "Je voudrais go to {dest}",
    "Train de {dep} to {dest}",
    "From {dep} a {dest}",
    "I want partir de {dep}",
    "Je veux a train to {dest}",
    "Please de {dep} a {dest}",
]

# =============================================================================
# NOT_TRIP TEMPLATES - FRENCH (100+)
# =============================================================================

# Service questions
NOT_TRIP_TEMPLATES_FR_SERVICE = [
    "Le train est en retard",
    "Mon train a du retard",
    "Ou sont les toilettes",
    "A quelle voie",
    "Quelle voie pour le train",
    "Ou est le guichet",
    "Ou acheter un billet",
    "Comment valider mon billet",
    "Ou est la sortie",
    "Ou est l'entree",
    "Y a-t-il un restaurant",
    "Ou manger dans la gare",
    "Ou est le cafe",
    "Ou attendre le train",
    "Quand arrive le prochain train",
    "Le train est annule",
    "Mon train est supprime",
    "Je suis perdu",
    "Pouvez-vous m'aider",
    "J'ai perdu mon billet",
    "J'ai oublie mes bagages",
    "Ou sont les consignes",
    "Ou deposer mes valises",
    "Wifi gratuit",
    "Y a-t-il du wifi",
]

# Chitchat/Social
NOT_TRIP_TEMPLATES_FR_CHITCHAT = [
    "Bonjour",
    "Salut",
    "Bonsoir",
    "Coucou",
    "Hello",
    "Hey",
    "Il fait beau aujourd'hui",
    "Il pleut dehors",
    "Quel temps fait-il",
    "T'as vu le match hier",
    "C'etait bien mon week-end",
    "Comment ca va",
    "Ca va bien",
    "Tres bien merci",
    "Quoi de neuf",
    "Pas grand chose",
    "Bonne journee",
    "Bonne soiree",
    "A bientot",
    "Au revoir",
    "Merci",
    "Merci beaucoup",
    "De rien",
    "Je vous en prie",
    "Pardon",
    "Excuse-moi",
    "Desole",
]

# General statements
NOT_TRIP_TEMPLATES_FR_GENERAL = [
    "J'aime les trains",
    "Les trains sont souvent en retard",
    "La SNCF c'est pas mal",
    "Je n'aime pas voyager",
    "C'est trop cher",
    "Les billets sont chers",
    "Le train c'est mieux que l'avion",
    "Je prefere le train",
    "Le TGV va vite",
    "J'adore voyager",
    "Le voyage c'est la vie",
    "Vive le train",
    "A bas les retards",
    "Pourquoi c'est si cher",
    "C'est scandaleux",
    "C'est nul",
    "C'est super",
    "Genial",
    "Ok",
    "D'accord",
    "Oui",
    "Non",
    "Peut-etre",
    "Je ne sais pas",
    "Aucune idee",
]

# Incomplete travel (could be confused)
NOT_TRIP_TEMPLATES_FR_INCOMPLETE = [
    "Je voudrais",
    "Je veux aller",
    "Un train",
    "Voyage",
    "Partir",
    "Arriver",
    "Demain",
    "Aujourd'hui",
    "Un billet",
    "Je pars",
    "Je voyage",
    "Aller",
    "Direction",
    "Destination",
    "Gare",
    "Station",
    "Quai",
    "Voie",
    "Trajet",
    "Itineraire",
    "TGV",
    "TER",
    "Intercites",
    "Train",
    "Reservation",
]

# Ambiguous with station names (trap sentences)
NOT_TRIP_TEMPLATES_FR_AMBIGUOUS = [
    "Je m'appelle Paris",
    "Mon ami Lyon va bien",
    "J'ai rencontre Marseille hier",
    "Albert est mon cousin",
    "Nice de te rencontrer",
    "Nancy est ma soeur",
    "Troyes est un prenom original",
    "J'ai mange a Tours hier",
    "Mon chat s'appelle Brest",
    "Orleans est une belle ville",
    "J'habite a Nantes depuis toujours",
    "Rennes me manque",
    "Je suis ne a Lille",
    "Ma grand-mere vit a Bordeaux",
    "Mon frere travaille a Toulouse",
    "Dijon c'est la moutarde",
    "Le vin de Bordeaux est bon",
    "La mer a Nice est belle",
    "Il neige a Grenoble",
    "Strasbourg et l'Alsace",
]

# Questions about the service/system
NOT_TRIP_TEMPLATES_FR_QUESTIONS = [
    "Comment ca marche",
    "Qu'est-ce que tu peux faire",
    "Aide moi",
    "Help",
    "C'est quoi ce service",
    "Tu fais quoi",
    "Tu es qui",
    "Qui es-tu",
    "Tu es un robot",
    "Tu es une IA",
    "Comment tu t'appelles",
    "Quel est ton nom",
    "Tu parles anglais",
    "Tu comprends",
    "Tu m'entends",
    "Repete",
    "Je n'ai pas compris",
    "Parle plus fort",
    "Plus lentement",
    "Quoi",
]

# Combine all French NOT_TRIP templates
NOT_TRIP_TEMPLATES_FR = (
    NOT_TRIP_TEMPLATES_FR_SERVICE
    + NOT_TRIP_TEMPLATES_FR_CHITCHAT
    + NOT_TRIP_TEMPLATES_FR_GENERAL
    + NOT_TRIP_TEMPLATES_FR_INCOMPLETE
    + NOT_TRIP_TEMPLATES_FR_AMBIGUOUS
    + NOT_TRIP_TEMPLATES_FR_QUESTIONS
)

# =============================================================================
# NOT_TRIP TEMPLATES - OTHER LANGUAGES
# =============================================================================

NOT_TRIP_TEMPLATES_EN = [
    "Hello",
    "Hi",
    "Good morning",
    "Good evening",
    "Thank you",
    "Thanks",
    "Please",
    "Sorry",
    "Excuse me",
    "What time is it",
    "How does this work",
    "Can you help me",
    "I don't understand",
    "Where is the exit",
    "Goodbye",
]

NOT_TRIP_TEMPLATES_ES = [
    "Hola",
    "Buenos dias",
    "Gracias",
    "Por favor",
    "Que hora es",
    "No entiendo",
    "Adios",
    "Hasta luego",
]

NOT_TRIP_TEMPLATES_DE = [
    "Guten Tag",
    "Hallo",
    "Danke",
    "Bitte",
    "Wie funktioniert das",
    "Ich verstehe nicht",
    "Auf Wiedersehen",
    "Tschuss",
]

NOT_TRIP_TEMPLATES_IT = [
    "Ciao",
    "Buongiorno",
    "Grazie",
    "Per favore",
    "Come funziona",
    "Non capisco",
    "Arrivederci",
]

# =============================================================================
# UNKNOWN TEMPLATES (50+)
# =============================================================================

UNKNOWN_TEMPLATES = [
    # Gibberish
    "asdfghjkl",
    "qwerty",
    "123456",
    "???",
    "...",
    "!!!",
    "@#$%",
    "aaaaaaaa",
    "test test test",
    "blablabla",
    "lorem ipsum",
    "",
    "   ",
    "a",
    "ab",
    "xyz",
    "mmmmm",
    "zzzzz",
    "lalala",
    "nanana",
    # Noise markers
    "[inaudible]",
    "[bruit]",
    "[bruit de fond]",
    "[musique]",
    "[static]",
    "[coupure]",
    "*bruit*",
    "---",
    "____",
    "[silence]",
    "[???]",
    # Whisper hallucinations
    "Merci d'avoir regarde cette video",
    "N'oubliez pas de vous abonner",
    "Like and subscribe",
    "Sous-titres par la communaute",
    "Copyright 2024",
    "Thanks for watching",
    "Abonnez-vous a notre chaine",
    "Rendez-vous sur notre site",
    "Pour plus d'informations",
    "Visitez notre site web",
    # Truncated
    "Je voudrais aller de Par...",
    "Un train de...",
    "De Lyo...",
    "Je veux all...",
    "Billet pour...",
    "Vers Ma...",
    "[coupure] de Paris a",
    "... Lyon",
    "... et voila",
    "puis... [coupure]",
]

# =============================================================================
# DATASET GENERATOR CLASS
# =============================================================================


class STTDatasetGenerator:
    """Generator for 100k STT simulation dataset."""

    def __init__(
        self,
        station_db: StationDatabase,
        seed: int | None = None,
    ):
        """Initialize the generator."""
        self.station_db = station_db
        self.stations = [s.name for s in station_db.get_all_stations(passenger_only=True)]
        self.major_stations = [
            s.name for s in station_db.get_all_stations() if s.short_code
        ]

        # Create augmenters for different intensities
        self.augmenters = {
            "clean": create_augmenter("clean", seed),
            "light": create_augmenter("light", seed),
            "moderate": create_augmenter("moderate", seed),
            "heavy": create_augmenter("heavy", seed),
        }

        if seed is not None:
            random.seed(seed)

        self.entry_count = 0

    def _get_id(self) -> str:
        """Generate unique sentence ID."""
        self.entry_count += 1
        return f"STT{self.entry_count:06d}"

    def _random_station(self, exclude: list[str] | None = None) -> str:
        """Get a random station name."""
        pool = self.major_stations if random.random() > 0.3 else self.stations
        if exclude:
            pool = [s for s in pool if s not in exclude]
        return random.choice(pool) if pool else random.choice(self.stations)

    def _pick_intensity(self) -> str:
        """Pick STT error intensity based on distribution."""
        r = random.random()
        if r < 0.20:
            return "clean"
        elif r < 0.60:
            return "light"
        elif r < 0.90:
            return "moderate"
        else:
            return "heavy"

    def _generate_trip_fr(self) -> DatasetEntry:
        """Generate a French TRIP entry."""
        dep = self._random_station()
        dest = self._random_station(exclude=[dep])

        template = random.choice(TRIP_TEMPLATES_FR)

        # Check if it's destination-only template
        if "{dep}" not in template:
            sentence = template.format(dest=dest)
            departure = ""
        else:
            sentence = template.format(dep=dep, dest=dest)
            departure = dep

        # Apply STT augmentation
        intensity = self._pick_intensity()
        augmenter = self.augmenters[intensity]

        # Optionally corrupt station names in sentence
        station_to_corrupt = dep if departure else dest
        sentence = augmenter.augment(
            sentence, apply_to_station=station_to_corrupt if random.random() < 0.3 else None
        )

        return DatasetEntry(
            sentence_id=self._get_id(),
            sentence=sentence,
            intent="TRIP",
            language="FRENCH",
            departure=departure,
            destination=dest,
        )

    def _generate_trip_en(self) -> DatasetEntry:
        """Generate an English TRIP entry."""
        dep = self._random_station()
        dest = self._random_station(exclude=[dep])
        template = random.choice(TRIP_TEMPLATES_EN)
        sentence = template.format(dep=dep, dest=dest)

        return DatasetEntry(
            sentence_id=self._get_id(),
            sentence=sentence,
            intent="TRIP",
            language="ENGLISH",
            departure=dep,
            destination=dest,
        )

    def _generate_trip_es(self) -> DatasetEntry:
        """Generate a Spanish TRIP entry."""
        dep = self._random_station()
        dest = self._random_station(exclude=[dep])
        template = random.choice(TRIP_TEMPLATES_ES)
        sentence = template.format(dep=dep, dest=dest)

        return DatasetEntry(
            sentence_id=self._get_id(),
            sentence=sentence,
            intent="TRIP",
            language="SPANISH",
            departure=dep,
            destination=dest,
        )

    def _generate_trip_de(self) -> DatasetEntry:
        """Generate a German TRIP entry."""
        dep = self._random_station()
        dest = self._random_station(exclude=[dep])
        template = random.choice(TRIP_TEMPLATES_DE)
        sentence = template.format(dep=dep, dest=dest)

        return DatasetEntry(
            sentence_id=self._get_id(),
            sentence=sentence,
            intent="TRIP",
            language="GERMAN",
            departure=dep,
            destination=dest,
        )

    def _generate_trip_it(self) -> DatasetEntry:
        """Generate an Italian TRIP entry."""
        dep = self._random_station()
        dest = self._random_station(exclude=[dep])
        template = random.choice(TRIP_TEMPLATES_IT)
        sentence = template.format(dep=dep, dest=dest)

        return DatasetEntry(
            sentence_id=self._get_id(),
            sentence=sentence,
            intent="TRIP",
            language="ITALIAN",
            departure=dep,
            destination=dest,
        )

    def _generate_trip_mixed(self) -> DatasetEntry:
        """Generate a mixed language TRIP entry."""
        dep = self._random_station()
        dest = self._random_station(exclude=[dep])
        template = random.choice(TRIP_TEMPLATES_MIXED)

        if "{dep}" in template:
            sentence = template.format(dep=dep, dest=dest)
            departure = dep
        else:
            sentence = template.format(dest=dest)
            departure = ""

        return DatasetEntry(
            sentence_id=self._get_id(),
            sentence=sentence,
            intent="TRIP",
            language="UNKNOWN",
            departure=departure,
            destination=dest,
        )

    # -------------------------------------------------------------------------
    # Intermediate stop generation methods
    # -------------------------------------------------------------------------

    def _generate_trip_fr_intermediate(self) -> DatasetEntry:
        """Generate a French TRIP entry with intermediate stop."""
        dep = self._random_station()
        via = self._random_station(exclude=[dep])
        dest = self._random_station(exclude=[dep, via])

        template = random.choice(TRIP_INTERMEDIATE_TEMPLATES_FR)
        sentence = template.format(dep=dep, dest=dest, via=via)

        # Apply STT augmentation
        intensity = self._pick_intensity()
        augmenter = self.augmenters[intensity]

        # Optionally corrupt station names in sentence
        station_to_corrupt = random.choice([dep, via, dest])
        sentence = augmenter.augment(
            sentence, apply_to_station=station_to_corrupt if random.random() < 0.3 else None
        )

        return DatasetEntry(
            sentence_id=self._get_id(),
            sentence=sentence,
            intent="TRIP",
            language="FRENCH",
            departure=dep,
            destination=dest,
            intermediate=via,
        )

    def _generate_trip_en_intermediate(self) -> DatasetEntry:
        """Generate an English TRIP entry with intermediate stop."""
        dep = self._random_station()
        via = self._random_station(exclude=[dep])
        dest = self._random_station(exclude=[dep, via])

        template = random.choice(TRIP_INTERMEDIATE_TEMPLATES_EN)
        sentence = template.format(dep=dep, dest=dest, via=via)

        return DatasetEntry(
            sentence_id=self._get_id(),
            sentence=sentence,
            intent="TRIP",
            language="ENGLISH",
            departure=dep,
            destination=dest,
            intermediate=via,
        )

    def _generate_trip_es_intermediate(self) -> DatasetEntry:
        """Generate a Spanish TRIP entry with intermediate stop."""
        dep = self._random_station()
        via = self._random_station(exclude=[dep])
        dest = self._random_station(exclude=[dep, via])

        template = random.choice(TRIP_INTERMEDIATE_TEMPLATES_ES)
        sentence = template.format(dep=dep, dest=dest, via=via)

        return DatasetEntry(
            sentence_id=self._get_id(),
            sentence=sentence,
            intent="TRIP",
            language="SPANISH",
            departure=dep,
            destination=dest,
            intermediate=via,
        )

    def _generate_trip_de_intermediate(self) -> DatasetEntry:
        """Generate a German TRIP entry with intermediate stop."""
        dep = self._random_station()
        via = self._random_station(exclude=[dep])
        dest = self._random_station(exclude=[dep, via])

        template = random.choice(TRIP_INTERMEDIATE_TEMPLATES_DE)
        sentence = template.format(dep=dep, dest=dest, via=via)

        return DatasetEntry(
            sentence_id=self._get_id(),
            sentence=sentence,
            intent="TRIP",
            language="GERMAN",
            departure=dep,
            destination=dest,
            intermediate=via,
        )

    def _generate_trip_it_intermediate(self) -> DatasetEntry:
        """Generate an Italian TRIP entry with intermediate stop."""
        dep = self._random_station()
        via = self._random_station(exclude=[dep])
        dest = self._random_station(exclude=[dep, via])

        template = random.choice(TRIP_INTERMEDIATE_TEMPLATES_IT)
        sentence = template.format(dep=dep, dest=dest, via=via)

        return DatasetEntry(
            sentence_id=self._get_id(),
            sentence=sentence,
            intent="TRIP",
            language="ITALIAN",
            departure=dep,
            destination=dest,
            intermediate=via,
        )

    def _generate_not_trip_fr(self) -> DatasetEntry:
        """Generate a French NOT_TRIP entry."""
        template = random.choice(NOT_TRIP_TEMPLATES_FR)

        # Some templates have {station} placeholder
        if "{station}" in template:
            station = self._random_station()
            sentence = template.format(station=station)
        else:
            sentence = template

        # Apply light STT augmentation to some
        if random.random() < 0.3:
            intensity = random.choice(["light", "moderate"])
            sentence = self.augmenters[intensity].augment(sentence)

        return DatasetEntry(
            sentence_id=self._get_id(),
            sentence=sentence,
            intent="NOT_TRIP",
            language="FRENCH",
        )

    def _generate_not_trip_en(self) -> DatasetEntry:
        """Generate an English NOT_TRIP entry."""
        sentence = random.choice(NOT_TRIP_TEMPLATES_EN)
        return DatasetEntry(
            sentence_id=self._get_id(),
            sentence=sentence,
            intent="NOT_TRIP",
            language="ENGLISH",
        )

    def _generate_not_trip_es(self) -> DatasetEntry:
        """Generate a Spanish NOT_TRIP entry."""
        sentence = random.choice(NOT_TRIP_TEMPLATES_ES)
        return DatasetEntry(
            sentence_id=self._get_id(),
            sentence=sentence,
            intent="NOT_TRIP",
            language="SPANISH",
        )

    def _generate_not_trip_de(self) -> DatasetEntry:
        """Generate a German NOT_TRIP entry."""
        sentence = random.choice(NOT_TRIP_TEMPLATES_DE)
        return DatasetEntry(
            sentence_id=self._get_id(),
            sentence=sentence,
            intent="NOT_TRIP",
            language="GERMAN",
        )

    def _generate_not_trip_it(self) -> DatasetEntry:
        """Generate an Italian NOT_TRIP entry."""
        sentence = random.choice(NOT_TRIP_TEMPLATES_IT)
        return DatasetEntry(
            sentence_id=self._get_id(),
            sentence=sentence,
            intent="NOT_TRIP",
            language="ITALIAN",
        )

    def _generate_unknown(self) -> DatasetEntry:
        """Generate an UNKNOWN intent entry."""
        sentence = random.choice(UNKNOWN_TEMPLATES)
        return DatasetEntry(
            sentence_id=self._get_id(),
            sentence=sentence,
            intent="UNKNOWN",
            language="UNKNOWN",
        )

    def generate_dataset(
        self, total: int = 100000, intermediate_ratio: float = 0.15
    ) -> list[DatasetEntry]:
        """
        Generate complete dataset with specified distribution.

        Distribution:
        - TRIP: 70% (FR: 56k, EN: 7k, ES: 2.5k, DE: 2.5k, IT: 1k, Mixed: 1k)
          - Of which ~15% have intermediate stops
        - NOT_TRIP: 25% (FR: 20k, EN: 3k, ES: 0.75k, DE: 0.75k, IT: 0.5k)
        - UNKNOWN: 5% (all UNKNOWN language)

        Args:
            total: Total number of entries to generate
            intermediate_ratio: Ratio of TRIP entries with intermediate stops (default: 15%)
        """
        entries: list[DatasetEntry] = []

        # Calculate counts
        trip_count = int(total * 0.70)
        not_trip_count = int(total * 0.25)
        unknown_count = total - trip_count - not_trip_count

        # TRIP distribution by language
        trip_fr = int(trip_count * 0.80)  # 80% French
        trip_en = int(trip_count * 0.10)  # 10% English
        trip_es = int(trip_count * 0.0357)  # ~3.57% Spanish
        trip_de = int(trip_count * 0.0357)  # ~3.57% German
        trip_it = int(trip_count * 0.0143)  # ~1.43% Italian
        trip_mixed = trip_count - trip_fr - trip_en - trip_es - trip_de - trip_it

        # Split each language into regular and intermediate
        trip_fr_intermediate = int(trip_fr * intermediate_ratio)
        trip_fr_regular = trip_fr - trip_fr_intermediate

        trip_en_intermediate = int(trip_en * intermediate_ratio)
        trip_en_regular = trip_en - trip_en_intermediate

        trip_es_intermediate = int(trip_es * intermediate_ratio)
        trip_es_regular = trip_es - trip_es_intermediate

        trip_de_intermediate = int(trip_de * intermediate_ratio)
        trip_de_regular = trip_de - trip_de_intermediate

        trip_it_intermediate = int(trip_it * intermediate_ratio)
        trip_it_regular = trip_it - trip_it_intermediate

        # NOT_TRIP distribution by language
        not_trip_fr = int(not_trip_count * 0.80)
        not_trip_en = int(not_trip_count * 0.12)
        not_trip_es = int(not_trip_count * 0.03)
        not_trip_de = int(not_trip_count * 0.03)
        not_trip_it = not_trip_count - not_trip_fr - not_trip_en - not_trip_es - not_trip_de

        # Generate French TRIP entries (regular + intermediate)
        print(f"Generating {trip_fr_regular} French TRIP entries (regular)...")
        for _ in range(trip_fr_regular):
            entries.append(self._generate_trip_fr())

        print(f"Generating {trip_fr_intermediate} French TRIP entries (intermediate)...")
        for _ in range(trip_fr_intermediate):
            entries.append(self._generate_trip_fr_intermediate())

        # Generate English TRIP entries (regular + intermediate)
        print(f"Generating {trip_en_regular} English TRIP entries (regular)...")
        for _ in range(trip_en_regular):
            entries.append(self._generate_trip_en())

        print(f"Generating {trip_en_intermediate} English TRIP entries (intermediate)...")
        for _ in range(trip_en_intermediate):
            entries.append(self._generate_trip_en_intermediate())

        # Generate Spanish TRIP entries (regular + intermediate)
        print(f"Generating {trip_es_regular} Spanish TRIP entries (regular)...")
        for _ in range(trip_es_regular):
            entries.append(self._generate_trip_es())

        print(f"Generating {trip_es_intermediate} Spanish TRIP entries (intermediate)...")
        for _ in range(trip_es_intermediate):
            entries.append(self._generate_trip_es_intermediate())

        # Generate German TRIP entries (regular + intermediate)
        print(f"Generating {trip_de_regular} German TRIP entries (regular)...")
        for _ in range(trip_de_regular):
            entries.append(self._generate_trip_de())

        print(f"Generating {trip_de_intermediate} German TRIP entries (intermediate)...")
        for _ in range(trip_de_intermediate):
            entries.append(self._generate_trip_de_intermediate())

        # Generate Italian TRIP entries (regular + intermediate)
        print(f"Generating {trip_it_regular} Italian TRIP entries (regular)...")
        for _ in range(trip_it_regular):
            entries.append(self._generate_trip_it())

        print(f"Generating {trip_it_intermediate} Italian TRIP entries (intermediate)...")
        for _ in range(trip_it_intermediate):
            entries.append(self._generate_trip_it_intermediate())

        # Generate Mixed TRIP entries (no intermediate for mixed language)
        print(f"Generating {trip_mixed} Mixed TRIP entries...")
        for _ in range(trip_mixed):
            entries.append(self._generate_trip_mixed())

        # Generate NOT_TRIP entries
        print(f"Generating {not_trip_fr} French NOT_TRIP entries...")
        for _ in range(not_trip_fr):
            entries.append(self._generate_not_trip_fr())

        print(f"Generating {not_trip_en} English NOT_TRIP entries...")
        for _ in range(not_trip_en):
            entries.append(self._generate_not_trip_en())

        print(f"Generating {not_trip_es} Spanish NOT_TRIP entries...")
        for _ in range(not_trip_es):
            entries.append(self._generate_not_trip_es())

        print(f"Generating {not_trip_de} German NOT_TRIP entries...")
        for _ in range(not_trip_de):
            entries.append(self._generate_not_trip_de())

        print(f"Generating {not_trip_it} Italian NOT_TRIP entries...")
        for _ in range(not_trip_it):
            entries.append(self._generate_not_trip_it())

        # Generate UNKNOWN entries
        print(f"Generating {unknown_count} UNKNOWN entries...")
        for _ in range(unknown_count):
            entries.append(self._generate_unknown())

        # Shuffle the dataset
        print("Shuffling dataset...")
        random.shuffle(entries)

        return entries


# =============================================================================
# DATASET EXPORT FUNCTIONS
# =============================================================================


def stratified_split(
    entries: list[DatasetEntry],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[list[DatasetEntry], list[DatasetEntry], list[DatasetEntry]]:
    """
    Split dataset into train/val/test with stratification by intent and language.
    """
    # Group by (intent, language)
    groups: dict[tuple[str, str], list[DatasetEntry]] = {}
    for entry in entries:
        key = (entry.intent, entry.language)
        if key not in groups:
            groups[key] = []
        groups[key].append(entry)

    train, val, test = [], [], []

    for key, group_entries in groups.items():
        random.shuffle(group_entries)
        n = len(group_entries)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train.extend(group_entries[:train_end])
        val.extend(group_entries[train_end:val_end])
        test.extend(group_entries[val_end:])

    # Shuffle each split
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def export_to_csv(entries: list[DatasetEntry], filepath: Path) -> None:
    """Export dataset to CSV file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sentence_id",
                "sentence",
                "intent",
                "language",
                "departure",
                "destination",
                "intermediate",
            ],
        )
        writer.writeheader()
        for entry in entries:
            writer.writerow(entry.to_dict())


def export_to_json(entries: list[DatasetEntry], filepath: Path) -> None:
    """Export dataset to JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump([e.to_dict() for e in entries], f, ensure_ascii=False, indent=2)


def print_stats(entries: list[DatasetEntry], name: str = "Dataset") -> None:
    """Print dataset statistics."""
    intent_counts: dict[str, int] = {}
    language_counts: dict[str, int] = {}
    intent_language_counts: dict[tuple[str, str], int] = {}
    intermediate_count = 0
    trip_count = 0

    for entry in entries:
        intent_counts[entry.intent] = intent_counts.get(entry.intent, 0) + 1
        language_counts[entry.language] = language_counts.get(entry.language, 0) + 1
        key = (entry.intent, entry.language)
        intent_language_counts[key] = intent_language_counts.get(key, 0) + 1
        if entry.intent == "TRIP":
            trip_count += 1
            if entry.intermediate:
                intermediate_count += 1

    print(f"\n{'=' * 60}")
    print(f"{name} Statistics")
    print(f"{'=' * 60}")
    print(f"Total entries: {len(entries)}")

    print(f"\nIntent Distribution:")
    for intent, count in sorted(intent_counts.items()):
        pct = 100 * count / len(entries)
        print(f"  {intent}: {count} ({pct:.1f}%)")

    print(f"\nLanguage Distribution:")
    for lang, count in sorted(language_counts.items()):
        pct = 100 * count / len(entries)
        print(f"  {lang}: {count} ({pct:.1f}%)")

    print(f"\nIntent x Language:")
    for (intent, lang), count in sorted(intent_language_counts.items()):
        pct = 100 * count / len(entries)
        print(f"  {intent} / {lang}: {count} ({pct:.1f}%)")

    if trip_count > 0:
        print(f"\nIntermediate Stops (TRIP only):")
        pct = 100 * intermediate_count / trip_count
        print(f"  With intermediate: {intermediate_count} ({pct:.1f}% of TRIP)")
        print(f"  Without intermediate: {trip_count - intermediate_count} ({100 - pct:.1f}% of TRIP)")


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate 100k STT simulation dataset for travel order resolution."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: datasets/generated/)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100000,
        help="Total number of sentences (default: 100000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "json", "both"],
        default="both",
        help="Output format (default: both)",
    )

    args = parser.parse_args()

    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(__file__).parent.parent / "generated"

    # Load station database
    print("Loading station database...")
    station_db = StationDatabase()
    station_db.load()
    stats = station_db.stats()
    print(f"Loaded {stats['passenger_stations']} passenger stations.")

    # Generate dataset
    print(f"\nGenerating dataset with {args.count} entries...")
    generator = STTDatasetGenerator(station_db, seed=args.seed)
    entries = generator.generate_dataset(total=args.count)

    # Print full dataset stats
    print_stats(entries, "Full Dataset")

    # Split into train/val/test
    print("\nSplitting dataset (70/15/15)...")
    train, val, test = stratified_split(entries)

    print_stats(train, "Train Set")
    print_stats(val, "Validation Set")
    print_stats(test, "Test Set")

    # Export
    print(f"\nExporting to {output_dir}...")

    if args.format in ("csv", "both"):
        export_to_csv(entries, output_dir / "stt_dataset_100k.csv")
        export_to_csv(train, output_dir / "train.csv")
        export_to_csv(val, output_dir / "val.csv")
        export_to_csv(test, output_dir / "test.csv")
        print("  CSV files exported.")

    if args.format in ("json", "both"):
        export_to_json(entries, output_dir / "stt_dataset_100k.json")
        export_to_json(train, output_dir / "train.json")
        export_to_json(val, output_dir / "val.json")
        export_to_json(test, output_dir / "test.json")
        print("  JSON files exported.")

    print("\nDone!")


if __name__ == "__main__":
    main()
