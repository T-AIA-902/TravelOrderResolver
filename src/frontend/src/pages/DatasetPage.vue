<script setup lang="ts">
import {
  Database,
  BarChart2,
  Layers,
  Globe,
  MessageSquare,
  AlertTriangle,
  BookOpen,
  Cpu,
  FileText,
  Shuffle,
} from 'lucide-vue-next'

const datasetOverview = [
  { label: 'Entrees totales', value: '100 000', description: 'par dataset' },
  { label: 'Datasets', value: '2', description: 'Base + Augmented (STT)' },
  { label: 'Seed', value: '42', description: 'reproductibilite garantie' },
  { label: 'Generation', value: 'Jan. 2026', description: 'synthetique' },
]

const splitRows = [
  { split: 'Train', file: 'train.csv', samples: '69 998', percent: '70%' },
  { split: 'Validation', file: 'val.csv', samples: '14 997', percent: '15%' },
  { split: 'Test', file: 'test.csv', samples: '15 005', percent: '15%' },
  { split: 'Total', file: '*_dataset_100k.csv', samples: '100 000', percent: '100%' },
]

const intentDistribution = [
  { intent: 'TRIP', count: '74 000', percent: '74%', color: 'bg-emerald-500', description: 'Demandes de voyage (toutes langues)' },
  { intent: 'NOT_TRIP', count: '25 000', percent: '25%', color: 'bg-blue-500', description: 'Phrases non liees au voyage' },
  { intent: 'UNKNOWN', count: '1 000', percent: '1%', color: 'bg-gray-400', description: 'Bruit, gibberish, incomprehensible' },
]

const languageDistribution = [
  { language: 'Francais', count: '89 100', percent: '89.1%', color: 'bg-blue-600' },
  { language: 'Anglais', count: '6 440', percent: '6.4%', color: 'bg-red-500' },
  { language: 'Espagnol', count: '865', percent: '0.9%', color: 'bg-yellow-500' },
  { language: 'Allemand', count: '865', percent: '0.9%', color: 'bg-orange-500' },
  { language: 'Italien', count: '620', percent: '0.6%', color: 'bg-green-500' },
  { language: 'Inconnu', count: '2 110', percent: '2.1%', color: 'bg-gray-400' },
]

const examplesTRIP = [
  { source: 'Base', text: 'Quand part le prochain train de Viry-Noureuil a Montchanin', lang: 'FR' },
  { source: 'STT', text: 'quAnd paRt le proChaiN traiN de viry-nouReuil a montchAnin', lang: 'FR' },
  { source: 'Base', text: 'De Paris a Marseille en passant par Lyon', lang: 'FR' },
  { source: 'STT', text: 'euh de paris a marseille en passant par lion', lang: 'FR' },
  { source: 'Base', text: 'Belle-Isle - Begard to Aubrives with a stop at Sausset-les-Pins', lang: 'EN' },
  { source: 'Base', text: 'Direct Neussargues', lang: 'FR' },
]

const examplesNOT_TRIP = [
  { source: 'Base', text: 'Good evening', lang: 'EN' },
  { source: 'Base', text: "J'ai mange a Tours hier", lang: 'FR' },
  { source: 'Base', text: 'Bonjour, comment ca va ?', lang: 'FR' },
]

const examplesUNKNOWN = [
  { source: 'Base', text: 'puis... [coupure]', lang: '?' },
  { source: 'Base', text: 'Rendez-vous sur notre site', lang: '?' },
]

const sttErrorTypes = [
  { type: 'Mots de remplissage', prob: '15%', example: '"euh", "hum", "ben", "alors"' },
  { type: 'Faux departs', prob: '8%', example: '"Je veux... enfin je voudrais"' },
  { type: 'Confusions phonetiques', prob: '10%', example: '"trin" (train), "voyaje" (voyage)' },
  { type: 'Erreurs noms de gares', prob: '12%', example: '"Monparnasse", "Lion" (Lyon)' },
  { type: 'Erreurs capitalisation', prob: '25%', example: 'Tout en minuscules, majuscules aleatoires' },
  { type: 'Erreurs ponctuation', prob: '20%', example: 'Ponctuation manquante ou erronee' },
  { type: 'Hallucinations Whisper', prob: '1.5%', example: '"N\'oubliez pas de vous abonner"' },
  { type: 'Homophones', prob: '8%', example: '"a/a", "ou/ou", "vers/vert"' },
]

const sttIntensity = [
  { level: 'Clean', percent: '20%', errors: 'Aucune erreur', color: 'bg-emerald-100 text-emerald-700' },
  { level: 'Light', percent: '40%', errors: '1-2 types', color: 'bg-yellow-100 text-yellow-700' },
  { level: 'Moderate', percent: '30%', errors: '2-4 types', color: 'bg-orange-100 text-orange-700' },
  { level: 'Heavy', percent: '10%', errors: '4+ types', color: 'bg-red-100 text-red-700' },
]

const modelDataUsage = [
  { model: 'CamemBERT (intent)', dataset: 'Augmented (STT)', split: 'train.csv', task: 'Classification d\'intention' },
  { model: 'SpaCy (intent)', dataset: 'Augmented (STT)', split: 'train.csv', task: 'Classification d\'intention' },
  { model: 'Regex + Fuzzy (entity)', dataset: 'Base + Augmented', split: 'Regles statiques', task: 'Extraction d\'entites' },
  { model: 'Langdetect (language)', dataset: 'Augmented (STT)', split: 'test.csv', task: 'Detection de langue' },
]

const columns = [
  { name: 'sentence_id', type: 'string', description: 'Identifiant unique (BASE/STT + 6 chiffres)' },
  { name: 'sentence', type: 'string', description: 'Texte de la phrase' },
  { name: 'intent', type: 'string', description: 'TRIP, NOT_TRIP, UNKNOWN' },
  { name: 'language', type: 'string', description: 'FRENCH, ENGLISH, SPANISH, GERMAN, ITALIAN, UNKNOWN' },
  { name: 'departure', type: 'string', description: 'Gare de depart (vide si non applicable)' },
  { name: 'destination', type: 'string', description: 'Gare d\'arrivee (vide si non applicable)' },
  { name: 'intermediate', type: 'string', description: 'Gare intermediaire "via" (vide si non applicable)' },
]
</script>

<template>
  <div class="space-y-6 p-8">
    <!-- Header -->
    <div class="flex items-center gap-4">
      <div class="flex h-12 w-12 items-center justify-center rounded-xl bg-emerald-50">
        <Database class="h-6 w-6 text-emerald-600" />
      </div>
      <div>
        <h1 class="text-2xl font-bold tracking-tight text-gray-900">Dataset</h1>
        <p class="text-gray-500">Composition, distribution et exemples des datasets du projet</p>
      </div>
    </div>

    <!-- Overview stat cards -->
    <div class="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
      <div
        v-for="stat in datasetOverview"
        :key="stat.label"
        class="rounded-xl border border-gray-200 bg-white p-5 shadow-sm"
      >
        <p class="text-sm text-gray-500">{{ stat.label }}</p>
        <p class="mt-1 text-2xl font-bold text-gray-900">{{ stat.value }}</p>
        <p class="mt-0.5 text-xs text-gray-400">{{ stat.description }}</p>
      </div>
    </div>

    <!-- Dataset architecture -->
    <div class="rounded-xl border border-gray-200 bg-white shadow-sm">
      <div class="flex items-center gap-3 border-b border-gray-200 px-6 py-5">
        <Layers class="h-5 w-5 text-emerald-500" />
        <div>
          <h2 class="text-lg font-semibold text-gray-900">Architecture des datasets</h2>
          <p class="text-sm text-gray-500">Deux niveaux pour etudes d'ablation</p>
        </div>
      </div>
      <div class="grid grid-cols-1 gap-4 p-6 md:grid-cols-2">
        <!-- Base dataset -->
        <div class="rounded-lg border border-gray-100 bg-gray-50/50 p-5">
          <div class="mb-3 flex items-center gap-2">
            <FileText class="h-4 w-4 text-blue-500" />
            <h3 class="font-semibold text-gray-900">Dataset Base</h3>
          </div>
          <p class="mb-3 text-sm text-gray-600">
            Phrases <span class="font-medium text-gray-800">propres</span>, sans erreurs de transcription.
            Grammaticalement correctes et bien formatees.
          </p>
          <p class="text-xs text-gray-400">Repertoire : <code class="rounded bg-gray-100 px-1.5 py-0.5 font-mono text-gray-600">datasets/base/</code></p>
          <p class="mt-1 text-xs text-gray-400">Format ID : <code class="rounded bg-gray-100 px-1.5 py-0.5 font-mono text-gray-600">BASE000001</code></p>
        </div>
        <!-- Augmented dataset -->
        <div class="rounded-lg border border-gray-100 bg-gray-50/50 p-5">
          <div class="mb-3 flex items-center gap-2">
            <Shuffle class="h-4 w-4 text-purple-500" />
            <h3 class="font-semibold text-gray-900">Dataset Augmented (STT)</h3>
          </div>
          <p class="mb-3 text-sm text-gray-600">
            Avec <span class="font-medium text-gray-800">erreurs STT simulees</span> appliquees aux phrases du dataset base.
            Simule des transcriptions realistes de Whisper.
          </p>
          <p class="text-xs text-gray-400">Repertoire : <code class="rounded bg-gray-100 px-1.5 py-0.5 font-mono text-gray-600">datasets/augmented/</code></p>
          <p class="mt-1 text-xs text-gray-400">Format ID : <code class="rounded bg-gray-100 px-1.5 py-0.5 font-mono text-gray-600">STT000001</code></p>
        </div>
      </div>
    </div>

    <!-- Train / Val / Test split -->
    <div class="rounded-xl border border-gray-200 bg-white shadow-sm">
      <div class="flex items-center gap-3 border-b border-gray-200 px-6 py-5">
        <BarChart2 class="h-5 w-5 text-emerald-500" />
        <div>
          <h2 class="text-lg font-semibold text-gray-900">Repartition Train / Val / Test</h2>
          <p class="text-sm text-gray-500">Identique pour les deux datasets (base et augmented)</p>
        </div>
      </div>
      <div class="overflow-x-auto">
        <table class="w-full">
          <thead>
            <tr class="border-b border-gray-200 bg-gray-50/80">
              <th class="px-6 py-3.5 text-left text-sm font-medium text-gray-600">Split</th>
              <th class="px-6 py-3.5 text-left text-sm font-medium text-gray-600">Fichier</th>
              <th class="px-6 py-3.5 text-left text-sm font-medium text-gray-600">Echantillons</th>
              <th class="px-6 py-3.5 text-left text-sm font-medium text-gray-600">Proportion</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="(row, i) in splitRows"
              :key="i"
              class="border-b border-gray-100 transition-colors last:border-0 hover:bg-gray-50/50"
              :class="{ 'bg-gray-50/40 font-semibold': row.split === 'Total' }"
            >
              <td class="px-6 py-4 font-medium text-gray-900">{{ row.split }}</td>
              <td class="px-6 py-4">
                <code class="rounded bg-gray-100 px-2 py-0.5 font-mono text-sm text-gray-600">{{ row.file }}</code>
              </td>
              <td class="px-6 py-4 font-mono text-gray-600">{{ row.samples }}</td>
              <td class="px-6 py-4">
                <div class="flex items-center gap-3">
                  <div class="h-2 w-24 overflow-hidden rounded-full bg-gray-100">
                    <div
                      class="h-full rounded-full bg-emerald-500 transition-all"
                      :style="{ width: row.percent }"
                    />
                  </div>
                  <span class="font-mono text-sm text-gray-600">{{ row.percent }}</span>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Intent distribution -->
    <div class="rounded-xl border border-gray-200 bg-white shadow-sm">
      <div class="flex items-center gap-3 border-b border-gray-200 px-6 py-5">
        <MessageSquare class="h-5 w-5 text-emerald-500" />
        <div>
          <h2 class="text-lg font-semibold text-gray-900">Distribution par intention</h2>
          <p class="text-sm text-gray-500">Repartition des 3 classes d'intention</p>
        </div>
      </div>
      <div class="p-6">
        <!-- Visual bar -->
        <div class="mb-6 flex h-6 overflow-hidden rounded-full">
          <div
            v-for="item in intentDistribution"
            :key="item.intent"
            :class="item.color"
            :style="{ width: item.percent }"
            class="transition-all"
          />
        </div>
        <!-- Legend -->
        <div class="grid grid-cols-1 gap-4 sm:grid-cols-3">
          <div
            v-for="item in intentDistribution"
            :key="item.intent"
            class="rounded-lg border border-gray-100 p-4"
          >
            <div class="mb-2 flex items-center gap-2">
              <div :class="item.color" class="h-3 w-3 rounded-full" />
              <span class="font-mono text-sm font-semibold text-gray-900">{{ item.intent }}</span>
            </div>
            <p class="text-2xl font-bold text-gray-900">{{ item.count }}</p>
            <p class="text-sm text-gray-500">{{ item.percent }} &mdash; {{ item.description }}</p>
          </div>
        </div>
      </div>
    </div>

    <!-- Language distribution -->
    <div class="rounded-xl border border-gray-200 bg-white shadow-sm">
      <div class="flex items-center gap-3 border-b border-gray-200 px-6 py-5">
        <Globe class="h-5 w-5 text-emerald-500" />
        <div>
          <h2 class="text-lg font-semibold text-gray-900">Distribution par langue</h2>
          <p class="text-sm text-gray-500">Optimisee pour le systeme ferroviaire francais</p>
        </div>
      </div>
      <div class="overflow-x-auto">
        <table class="w-full">
          <thead>
            <tr class="border-b border-gray-200 bg-gray-50/80">
              <th class="px-6 py-3.5 text-left text-sm font-medium text-gray-600">Langue</th>
              <th class="px-6 py-3.5 text-left text-sm font-medium text-gray-600">Echantillons</th>
              <th class="px-6 py-3.5 text-left text-sm font-medium text-gray-600">Proportion</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="(row, i) in languageDistribution"
              :key="i"
              class="border-b border-gray-100 transition-colors last:border-0 hover:bg-gray-50/50"
            >
              <td class="px-6 py-4">
                <div class="flex items-center gap-2">
                  <div :class="row.color" class="h-2.5 w-2.5 rounded-full" />
                  <span class="font-medium text-gray-900">{{ row.language }}</span>
                </div>
              </td>
              <td class="px-6 py-4 font-mono text-gray-600">{{ row.count }}</td>
              <td class="px-6 py-4">
                <div class="flex items-center gap-3">
                  <div class="h-2 w-32 overflow-hidden rounded-full bg-gray-100">
                    <div
                      :class="row.color"
                      class="h-full rounded-full transition-all"
                      :style="{ width: row.percent }"
                    />
                  </div>
                  <span class="font-mono text-sm text-gray-600">{{ row.percent }}</span>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Example sentences -->
    <div class="rounded-xl border border-gray-200 bg-white shadow-sm">
      <div class="flex items-center gap-3 border-b border-gray-200 px-6 py-5">
        <BookOpen class="h-5 w-5 text-emerald-500" />
        <div>
          <h2 class="text-lg font-semibold text-gray-900">Exemples par categorie</h2>
          <p class="text-sm text-gray-500">Phrases representatives de chaque classe d'intention</p>
        </div>
      </div>
      <div class="space-y-0 divide-y divide-gray-100">
        <!-- TRIP examples -->
        <div class="p-6">
          <div class="mb-3 flex items-center gap-2">
            <span class="rounded-md bg-emerald-50 px-2.5 py-1 text-sm font-semibold text-emerald-700">TRIP</span>
            <span class="text-sm text-gray-400">Demandes de voyage</span>
          </div>
          <div class="space-y-2">
            <div
              v-for="(ex, i) in examplesTRIP"
              :key="i"
              class="flex items-start gap-3 rounded-lg bg-gray-50/70 px-4 py-3"
            >
              <span
                class="mt-0.5 shrink-0 rounded px-1.5 py-0.5 text-xs font-medium"
                :class="ex.source === 'STT' ? 'bg-purple-100 text-purple-700' : 'bg-blue-100 text-blue-700'"
              >
                {{ ex.source }}
              </span>
              <code class="text-sm text-gray-700">{{ ex.text }}</code>
              <span class="ml-auto shrink-0 rounded bg-gray-100 px-1.5 py-0.5 text-xs font-mono text-gray-500">{{ ex.lang }}</span>
            </div>
          </div>
        </div>
        <!-- NOT_TRIP examples -->
        <div class="p-6">
          <div class="mb-3 flex items-center gap-2">
            <span class="rounded-md bg-blue-50 px-2.5 py-1 text-sm font-semibold text-blue-700">NOT_TRIP</span>
            <span class="text-sm text-gray-400">Hors voyage</span>
          </div>
          <div class="space-y-2">
            <div
              v-for="(ex, i) in examplesNOT_TRIP"
              :key="i"
              class="flex items-start gap-3 rounded-lg bg-gray-50/70 px-4 py-3"
            >
              <span class="mt-0.5 shrink-0 rounded bg-blue-100 px-1.5 py-0.5 text-xs font-medium text-blue-700">
                {{ ex.source }}
              </span>
              <code class="text-sm text-gray-700">{{ ex.text }}</code>
              <span class="ml-auto shrink-0 rounded bg-gray-100 px-1.5 py-0.5 text-xs font-mono text-gray-500">{{ ex.lang }}</span>
            </div>
          </div>
        </div>
        <!-- UNKNOWN examples -->
        <div class="p-6">
          <div class="mb-3 flex items-center gap-2">
            <span class="rounded-md bg-gray-100 px-2.5 py-1 text-sm font-semibold text-gray-600">UNKNOWN</span>
            <span class="text-sm text-gray-400">Bruit / incomprehensible</span>
          </div>
          <div class="space-y-2">
            <div
              v-for="(ex, i) in examplesUNKNOWN"
              :key="i"
              class="flex items-start gap-3 rounded-lg bg-gray-50/70 px-4 py-3"
            >
              <span class="mt-0.5 shrink-0 rounded bg-gray-200 px-1.5 py-0.5 text-xs font-medium text-gray-600">
                {{ ex.source }}
              </span>
              <code class="text-sm text-gray-700">{{ ex.text }}</code>
              <span class="ml-auto shrink-0 rounded bg-gray-100 px-1.5 py-0.5 text-xs font-mono text-gray-500">{{ ex.lang }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- STT error simulation -->
    <div class="rounded-xl border border-gray-200 bg-white shadow-sm">
      <div class="flex items-center gap-3 border-b border-gray-200 px-6 py-5">
        <AlertTriangle class="h-5 w-5 text-amber-500" />
        <div>
          <h2 class="text-lg font-semibold text-gray-900">Erreurs STT simulees</h2>
          <p class="text-sm text-gray-500">14 types d'erreurs basees sur les comportements reels de Whisper</p>
        </div>
      </div>
      <!-- Intensity distribution -->
      <div class="border-b border-gray-100 p-6">
        <h3 class="mb-3 text-sm font-semibold text-gray-700">Distribution des intensites</h3>
        <div class="flex flex-wrap gap-3">
          <div
            v-for="level in sttIntensity"
            :key="level.level"
            :class="level.color"
            class="rounded-lg px-4 py-2.5"
          >
            <span class="text-sm font-semibold">{{ level.level }}</span>
            <span class="ml-2 text-sm opacity-80">{{ level.percent }}</span>
            <span class="ml-1 text-xs opacity-60">({{ level.errors }})</span>
          </div>
        </div>
      </div>
      <!-- Error types table -->
      <div class="overflow-x-auto">
        <table class="w-full">
          <thead>
            <tr class="border-b border-gray-200 bg-gray-50/80">
              <th class="px-6 py-3.5 text-left text-sm font-medium text-gray-600">Type d'erreur</th>
              <th class="px-6 py-3.5 text-left text-sm font-medium text-gray-600">Probabilite</th>
              <th class="px-6 py-3.5 text-left text-sm font-medium text-gray-600">Exemple</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="(row, i) in sttErrorTypes"
              :key="i"
              class="border-b border-gray-100 transition-colors last:border-0 hover:bg-gray-50/50"
            >
              <td class="px-6 py-4 font-medium text-gray-900">{{ row.type }}</td>
              <td class="px-6 py-4 font-mono text-gray-600">{{ row.prob }}</td>
              <td class="px-6 py-4 text-sm text-gray-500">{{ row.example }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Model data usage -->
    <div class="rounded-xl border border-gray-200 bg-white shadow-sm">
      <div class="flex items-center gap-3 border-b border-gray-200 px-6 py-5">
        <Cpu class="h-5 w-5 text-emerald-500" />
        <div>
          <h2 class="text-lg font-semibold text-gray-900">Utilisation par les modeles</h2>
          <p class="text-sm text-gray-500">Quel modele utilise quel dataset</p>
        </div>
      </div>
      <div class="overflow-x-auto">
        <table class="w-full">
          <thead>
            <tr class="border-b border-gray-200 bg-gray-50/80">
              <th class="px-6 py-3.5 text-left text-sm font-medium text-gray-600">Modele</th>
              <th class="px-6 py-3.5 text-left text-sm font-medium text-gray-600">Dataset</th>
              <th class="px-6 py-3.5 text-left text-sm font-medium text-gray-600">Split</th>
              <th class="px-6 py-3.5 text-left text-sm font-medium text-gray-600">Tache</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="(row, i) in modelDataUsage"
              :key="i"
              class="border-b border-gray-100 transition-colors last:border-0 hover:bg-gray-50/50"
            >
              <td class="px-6 py-4">
                <span class="rounded-md bg-emerald-50 px-2.5 py-1 text-sm font-medium text-emerald-700">
                  {{ row.model }}
                </span>
              </td>
              <td class="px-6 py-4 font-medium text-gray-900">{{ row.dataset }}</td>
              <td class="px-6 py-4">
                <code class="rounded bg-gray-100 px-2 py-0.5 font-mono text-sm text-gray-600">{{ row.split }}</code>
              </td>
              <td class="px-6 py-4 text-gray-600">{{ row.task }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Schema -->
    <div class="rounded-xl border border-gray-200 bg-white shadow-sm">
      <div class="flex items-center gap-3 border-b border-gray-200 px-6 py-5">
        <FileText class="h-5 w-5 text-emerald-500" />
        <div>
          <h2 class="text-lg font-semibold text-gray-900">Schema des colonnes</h2>
          <p class="text-sm text-gray-500">Structure commune aux deux datasets (CSV / JSON)</p>
        </div>
      </div>
      <div class="overflow-x-auto">
        <table class="w-full">
          <thead>
            <tr class="border-b border-gray-200 bg-gray-50/80">
              <th class="px-6 py-3.5 text-left text-sm font-medium text-gray-600">Colonne</th>
              <th class="px-6 py-3.5 text-left text-sm font-medium text-gray-600">Type</th>
              <th class="px-6 py-3.5 text-left text-sm font-medium text-gray-600">Description</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="(col, i) in columns"
              :key="i"
              class="border-b border-gray-100 transition-colors last:border-0 hover:bg-gray-50/50"
            >
              <td class="px-6 py-4">
                <code class="rounded bg-gray-100 px-2 py-0.5 font-mono text-sm font-medium text-gray-700">{{ col.name }}</code>
              </td>
              <td class="px-6 py-4 font-mono text-sm text-gray-500">{{ col.type }}</td>
              <td class="px-6 py-4 text-sm text-gray-600">{{ col.description }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>
