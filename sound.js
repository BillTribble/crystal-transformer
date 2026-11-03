// Enso Ambient Soundscape Engine for Crystal Transformer
// Faithfully ports the 6 modal Circle-of-Fifths ambient tracks, scales, phrases, and 6 voices
// (Felt Piano, CS-80 Ribbon Pad, DX7 Glass Bell, Spectral Choir, Revox Tape Loop, Sub-Bass Pedal)
// from Enso Soundscapes (//depot/google3/experimental/users/tribble/eno_soundscape).
//
// Acoustic & Generative Fidelity to Enso-Soundscapes:
// 1. Clean 320 kbps Salamander Grand Piano v3 samples (samples/piano_*.ogg + .mp3) baked with
//    Enso's 3-stage warm wool-felt lowpass filter (cutoff ~950 Hz, Q = 0.5) + smooth dark-velvet
//    stereo cathedral impulse response (lowpassed at 1350 Hz, Q = 0.5, ZERO +12st sine shimmer
//    on piano, ZERO comb-filter early-reflection spikes, ZERO slapback delay taps) + live per-track
//    Q = 0.5 felt lowpass bus so pitch-shifted notes never ring.
// 2. Faithful Enso voice balance (pianoCrossfade = -0.62 -> piano foreground 1.0, background
//    voices 0.38 * DEFAULT_VOICE_INSTRUMENT_TRIM), slow 2.5s–4.2s pad/choir/drone attack swells,
//    spacious 15s–24s Music-for-Airports breathing room between piano phrases with Enso's 3-minute
//    phrase-length LFO and aleatoric note/phrase gates, and a 2-voice background concurrency cap.
// 3. Automatic Circle-of-Fifths track rotation every 3 minutes (180 seconds).

export const TRACK_ROTATE_SECONDS = 180;

export const ENSO_REVERB_MODES = [
  'Stamped OGG (0% CPU)',
  'Hybrid (OGG + Lite FDN)',
  'Lite FDN Live',
];

export const ENSO_DRONE_MODES = [
  'Breathing wash',
  'Continuous drone',
];

export const ENSO_WASH_INSTRUMENTS = [
  'choir',
  'drone',
  'pad',
  'tape',
];

export const SAMPLE_ZONES = [
  { id: 'piano_c2', voice: 'piano', midi: 36, dur: 12.0 },
  { id: 'piano_fs2', voice: 'piano', midi: 42, dur: 12.0 },
  { id: 'piano_c3', voice: 'piano', midi: 48, dur: 11.8 },
  { id: 'piano_fs3', voice: 'piano', midi: 54, dur: 11.8 },
  { id: 'piano_c4', voice: 'piano', midi: 60, dur: 11.6 },
  { id: 'piano_fs4', voice: 'piano', midi: 66, dur: 11.5 },
  { id: 'piano_c5', voice: 'piano', midi: 72, dur: 11.4 },
  { id: 'piano_fs5', voice: 'piano', midi: 78, dur: 11.2 },
  { id: 'piano_c6', voice: 'piano', midi: 84, dur: 11.0 },

  { id: 'pad_c3', voice: 'pad', midi: 48, dur: 12.5 },
  { id: 'pad_g3', voice: 'pad', midi: 55, dur: 12.5 },
  { id: 'pad_c4', voice: 'pad', midi: 60, dur: 12.5 },
  { id: 'pad_g4', voice: 'pad', midi: 67, dur: 12.5 },

  { id: 'bell_c4', voice: 'bell', midi: 60, dur: 5.8 },
  { id: 'bell_g4', voice: 'bell', midi: 67, dur: 5.8 },
  { id: 'bell_c5', voice: 'bell', midi: 72, dur: 5.5 },
  { id: 'bell_g5', voice: 'bell', midi: 79, dur: 5.5 },

  { id: 'choir_c4', voice: 'choir', midi: 60, dur: 12.5 },
  { id: 'choir_g4', voice: 'choir', midi: 67, dur: 12.5 },
  { id: 'choir_c5', voice: 'choir', midi: 72, dur: 12.5 },

  { id: 'tape_c3', voice: 'tape', midi: 48, dur: 12.5 },
  { id: 'tape_g3', voice: 'tape', midi: 55, dur: 12.5 },
  { id: 'tape_c4', voice: 'tape', midi: 60, dur: 12.5 },

  { id: 'drone_c2', voice: 'drone', midi: 36, dur: 12.5 },
  { id: 'drone_g2', voice: 'drone', midi: 43, dur: 12.5 },
  { id: 'drone_c3', voice: 'drone', midi: 48, dur: 12.5 },
];

// Exact instrument trim & crossfade balance from eno_soundscape/js/audio-engine.js
// (pianoCrossfade = -0.62 -> pianoMultiplier = 1.0, otherMultiplier = 0.38)
const VOICE_INSTRUMENT_TRIM = {
  piano: 0.85,
  pad: 0.15,
  bell: 0.18,
  choir: 0.16,
  tape: 0.17,
  drone: 0.18,
};

export const ENSO_TRACKS = [
  {
    id: 'reflection',
    name: 'Luminous Drift',
    label: 'Luminous Drift (C Lydian)',
    keyLabel: 'C Lydian',
    tonic: 'C',
    rootMidi: 48,
    intervals: [0, 2, 4, 6, 7, 9, 11],
    degreeWeights: [0.98, 0.86, 0.42, 0.78, 0.92, 0.48, 0.32],
    markovGravity: [0.98, 0.86, 0.35, 0.78, 0.92, 0.42, 0.28],
    rotationSpeed: 1.00,
    pianoCutoffHz: 980,
    pianoShelfDb: -8.8,
    globalAttackSec: 4.2,
    phraseCycleSec: 17.6,
    phraseGateProb: 0.72,
    phraseLengthLfo: { enabled: false },
    droneMode: 'Breathing wash',
    washInstrument: 'choir',
    washFade: 4.2,
    washSustain: 7.0,
    washGap: 3.0,
    chordInstrument: 'choir',
    chordRollMs: 168,
    chordVel: 0.48,
    voices: {
      pad:   { cycle: 21.3, prob: 0.66, gain: 0.65, attack: 4.2, release: 9.5, octave: 0, pan: 0.38, muted: false },
      bell:  { cycle: 19.3, prob: 0.36, gain: 0.56, attack: 0.08, release: 5.5, octave: 1, pan: 0.60, muted: false },
      choir: { cycle: 29.9, prob: 0.48, gain: 0.60, attack: 4.5, release: 10.5, octave: 1, pan: 0.50, muted: false },
      tape:  { cycle: 23.7, prob: 0.46, gain: 0.58, attack: 3.0, release: 6.5, octave: 0, pan: 0.40, muted: false },
      drone: { cycle: 34.1, prob: 0.0,  gain: 0.65, attack: 4.6, release: 12.0, octave: -1, pan: 0.15, muted: true },
    },
    phrases: [
      // Lydian Glass Progression & Luminous Cell from eno_soundscape/js/phrase-patterns.js
      [
        { deg: [0, 2, 4, 6], oct: [0, 0, 0, 0], delayMs: 0, vel: 0.44, rollMs: 162 },
        { deg: [1, 3, 5, 0], oct: [0, 0, 0, 1], delayMs: 1950, vel: 0.38, rollMs: 178 },
        { deg: [6, 1, 3, 5], oct: [-1, 0, 0, 0], delayMs: 4100, vel: 0.35, rollMs: 192 },
        { deg: [0, 3, 4, 6], oct: [0, 0, 0, 0], delayMs: 6300, vel: 0.42, rollMs: 168 },
      ],
      [
        { deg: [0], oct: [1], delayMs: 0, vel: 0.44, rollMs: 155 },
        { deg: [3], oct: [1], delayMs: 840, vel: 0.36, rollMs: 155 },
        { deg: [4], oct: [1], delayMs: 1820, vel: 0.40, rollMs: 165 },
        { deg: [1], oct: [1], delayMs: 3040, vel: 0.34, rollMs: 175 },
      ],
    ],
    chords: [
      { deg: [0, 2, 4, 6], oct: [0, 0, 0, 0], rollMs: 162, vel: 0.46 },
      { deg: [1, 3, 5, 0], oct: [0, 0, 0, 1], rollMs: 178, vel: 0.40 },
      { deg: [6, 1, 3, 5], oct: [-1, 0, 0, 0], rollMs: 192, vel: 0.38 },
      { deg: [0, 3, 4, 6], oct: [0, 0, 0, 0], rollMs: 168, vel: 0.44 },
    ],
  },
  {
    id: 'thursday',
    name: 'Sunlit Atrium',
    label: 'Sunlit Atrium (G Mixolydian)',
    keyLabel: 'G Mixolydian',
    tonic: 'G',
    rootMidi: 55,
    intervals: [0, 2, 4, 5, 7, 9, 10],
    degreeWeights: [0.90, 0.75, 0.60, 0.30, 0.80, 0.50, 0.40],
    markovGravity: [0.92, 0.78, 0.72, 0.32, 0.88, 0.62, 0.76],
    rotationSpeed: 0.90,
    pianoCutoffHz: 1120,
    pianoShelfDb: -6.5,
    globalAttackSec: 3.2,
    phraseCycleSec: 15.8,
    phraseGateProb: 0.76,
    phraseLengthLfo: { enabled: true, periodSec: 180, minRatio: 0.50, maxRatio: 1.00 },
    droneMode: 'Breathing wash',
    washInstrument: 'drone',
    washFade: 4.3,
    washSustain: 7.0,
    washGap: 2.8,
    chordInstrument: 'choir',
    chordRollMs: 112,
    chordVel: 0.54,
    voices: {
      pad:   { cycle: 23.3, prob: 0.62, gain: 0.65, attack: 3.2, release: 8.0, octave: 0, pan: 0.35, muted: false },
      bell:  { cycle: 19.7, prob: 0.34, gain: 0.58, attack: 0.04, release: 4.8, octave: 1, pan: 0.55, muted: false },
      choir: { cycle: 18.7, prob: 0.72, gain: 0.62, attack: 3.4, release: 9.2, octave: 1, pan: 0.45, muted: false },
      tape:  { cycle: 27.9, prob: 0.42, gain: 0.58, attack: 2.4, release: 5.8, octave: 0, pan: 0.38, muted: false },
      drone: { cycle: 37.3, prob: 0.0,  gain: 0.65, attack: 3.8, release: 10.5, octave: -1, pan: 0.15, muted: true },
    },
    phrases: [
      // Sunlit Canopy Cascade (primarySteps & variantSteps from eno_soundscape/js/phrase-patterns.js)
      [
        { deg: [0, 4], oct: [0, 1], delayMs: 0, vel: 0.48, rollMs: 115 },
        { deg: [2], oct: [1], delayMs: 440, vel: 0.40, rollMs: 90 },
        { deg: [1], oct: [1], delayMs: 880, vel: 0.36, rollMs: 90 },
        { deg: [4], oct: [1], delayMs: 1320, vel: 0.44, rollMs: 95 },
        { deg: [6], oct: [0], delayMs: 1960, vel: 0.34, rollMs: 95 },
        { deg: [5], oct: [0], delayMs: 2420, vel: 0.38, rollMs: 95 },
        { deg: [2], oct: [0], delayMs: 2880, vel: 0.33, rollMs: 95 },
        { deg: [0], oct: [1], delayMs: 3640, vel: 0.45, rollMs: 115 },
      ],
      [
        { deg: [0, 4], oct: [0, 1], delayMs: 0, vel: 0.46, rollMs: 115 },
        { deg: [1], oct: [1], delayMs: 440, vel: 0.38, rollMs: 90 },
        { deg: [2], oct: [1], delayMs: 900, vel: 0.42, rollMs: 90 },
        { deg: [6], oct: [1], delayMs: 1380, vel: 0.35, rollMs: 95 },
        { deg: [4], oct: [1], delayMs: 2040, vel: 0.43, rollMs: 95 },
        { deg: [3], oct: [0], delayMs: 2500, vel: 0.34, rollMs: 95 },
        { deg: [1], oct: [0], delayMs: 2960, vel: 0.36, rollMs: 95 },
        { deg: [0], oct: [1], delayMs: 3720, vel: 0.44, rollMs: 115 },
      ],
    ],
    chords: [
      { deg: [0, 2, 4, 6], oct: [0, 0, 0, 0], rollMs: 112, vel: 0.52 },
      { deg: [6, 1, 3, 5], oct: [-1, 0, 0, 0], rollMs: 118, vel: 0.48 },
      { deg: [3, 5, 0, 2], oct: [0, 0, 1, 1], rollMs: 124, vel: 0.50 },
      { deg: [4, 6, 1, 3], oct: [0, 0, 1, 1], rollMs: 112, vel: 0.48 },
    ],
  },
  {
    id: 'installations',
    name: 'Cathedral Glass',
    label: 'Cathedral Glass (F Lydian)',
    keyLabel: 'F Lydian',
    tonic: 'F',
    rootMidi: 53,
    intervals: [0, 2, 4, 6, 7, 9, 11],
    degreeWeights: [0.85, 0.75, 0.70, 0.50, 0.90, 0.60, 0.35],
    markovGravity: [1.40, 1.32, 1.0, 1.44, 1.55, 1.0, 1.0],
    rotationSpeed: 0.80,
    pianoCutoffHz: 1140,
    pianoShelfDb: -5.5,
    globalAttackSec: 5.2,
    phraseCycleSec: 19.4,
    phraseGateProb: 0.70,
    phraseLengthLfo: { enabled: false },
    droneMode: 'Breathing wash',
    washInstrument: 'drone',
    washFade: 4.5,
    washSustain: 7.2,
    washGap: 3.2,
    chordInstrument: 'pad',
    chordRollMs: 208,
    chordVel: 0.48,
    voices: {
      pad:   { cycle: 26.9, prob: 0.68, gain: 0.65, attack: 5.0, release: 12.0, octave: 0, pan: -0.38, muted: false },
      bell:  { cycle: 16.9, prob: 0.42, gain: 0.58, attack: 0.12, release: 7.2, octave: 1, pan: 0.62, muted: false },
      choir: { cycle: 31.1, prob: 0.58, gain: 0.62, attack: 5.2, release: 13.5, octave: 1, pan: -0.60, muted: false },
      tape:  { cycle: 23.3, prob: 0.48, gain: 0.58, attack: 3.4, release: 8.5, octave: 0, pan: 0.48, muted: false },
      drone: { cycle: 41.1, prob: 0.0,  gain: 0.65, attack: 5.4, release: 15.0, octave: -1, pan: 0.18, muted: true },
    },
    phrases: [
      // Cathedral Vespers Chords & Gallery Horizon Pattern from eno_soundscape/js/phrase-patterns.js
      [
        { deg: [0, 2, 4, 1], oct: [0, 0, 0, 1], delayMs: 0, vel: 0.46, rollMs: 196 },
        { deg: [1, 3, 5, 0], oct: [0, 0, 0, 1], delayMs: 2150, vel: 0.40, rollMs: 216 },
        { deg: [5, 0, 2, 4], oct: [-1, 0, 0, 0], delayMs: 4400, vel: 0.36, rollMs: 204 },
        { deg: [4, 6, 1, 3], oct: [0, 0, 1, 1], delayMs: 6700, vel: 0.42, rollMs: 224 },
      ],
      [
        { deg: [0], oct: [1], delayMs: 0, vel: 0.46, rollMs: 185 },
        { deg: [2], oct: [1], delayMs: 950, vel: 0.38, rollMs: 185 },
        { deg: [3], oct: [1], delayMs: 2050, vel: 0.42, rollMs: 195 },
        { deg: [4], oct: [1], delayMs: 3350, vel: 0.36, rollMs: 210 },
      ],
    ],
    chords: [
      { deg: [0, 2, 4, 1], oct: [0, 0, 0, 1], rollMs: 196, vel: 0.46 },
      { deg: [1, 3, 5, 0], oct: [0, 0, 0, 1], rollMs: 216, vel: 0.40 },
      { deg: [5, 0, 2, 4], oct: [-1, 0, 0, 0], rollMs: 204, vel: 0.38 },
      { deg: [4, 6, 1, 3], oct: [0, 0, 1, 1], rollMs: 224, vel: 0.44 },
    ],
  },
  {
    id: 'apollo_shimmer',
    name: 'Weightless Orbit',
    label: 'Weightless Orbit (C Major)',
    keyLabel: 'C Major',
    tonic: 'C',
    rootMidi: 48,
    intervals: [0, 2, 4, 5, 7, 9, 11],
    degreeWeights: [0.90, 0.82, 0.65, 0.20, 0.85, 0.70, 0.30],
    markovGravity: [1.55, 1.42, 1.32, 1.0, 1.60, 1.35, 1.0],
    rotationSpeed: 0.74,
    pianoCutoffHz: 1080,
    pianoShelfDb: -7.2,
    globalAttackSec: 4.8,
    phraseCycleSec: 17.2,
    phraseGateProb: 0.74,
    phraseLengthLfo: { enabled: false },
    droneMode: 'Breathing wash',
    washInstrument: 'tape',
    washFade: 4.0,
    washSustain: 6.8,
    washGap: 2.8,
    chordInstrument: 'pad',
    chordRollMs: 152,
    chordVel: 0.50,
    voices: {
      pad:   { cycle: 21.1, prob: 0.74, gain: 0.65, attack: 4.8, release: 12.5, octave: 0, pan: 0.36, muted: false },
      bell:  { cycle: 22.9, prob: 0.44, gain: 0.58, attack: 0.08, release: 7.5, octave: 1, pan: 0.60, muted: false },
      choir: { cycle: 27.7, prob: 0.58, gain: 0.62, attack: 4.8, release: 14.0, octave: 1, pan: 0.50, muted: false },
      tape:  { cycle: 24.3, prob: 0.44, gain: 0.56, attack: 3.2, release: 8.0, octave: 0, pan: 0.40, muted: false },
      drone: { cycle: 37.9, prob: 0.0,  gain: 0.65, attack: 5.0, release: 15.5, octave: -1, pan: 0.15, muted: true },
    },
    phrases: [
      // Tranquility Base Sequence from eno_soundscape/js/phrase-patterns.js
      [
        { deg: [0, 4], oct: [0, 1], delayMs: 0, vel: 0.48, rollMs: 150 },
        { deg: [1], oct: [1], delayMs: 650, vel: 0.37, rollMs: 135 },
        { deg: [2], oct: [1], delayMs: 1380, vel: 0.41, rollMs: 135 },
        { deg: [4], oct: [1], delayMs: 2180, vel: 0.45, rollMs: 145 },
        { deg: [5], oct: [1], delayMs: 3360, vel: 0.38, rollMs: 150 },
        { deg: [4], oct: [1], delayMs: 4120, vel: 0.42, rollMs: 140 },
        { deg: [2], oct: [1], delayMs: 4940, vel: 0.35, rollMs: 135 },
        { deg: [0], oct: [1], delayMs: 6040, vel: 0.46, rollMs: 155 },
      ],
      [
        { deg: [4], oct: [1], delayMs: 0, vel: 0.45, rollMs: 145 },
        { deg: [1], oct: [1], delayMs: 740, vel: 0.37, rollMs: 145 },
        { deg: [2], oct: [1], delayMs: 1620, vel: 0.41, rollMs: 155 },
        { deg: [0], oct: [1], delayMs: 2820, vel: 0.46, rollMs: 165 },
      ],
    ],
    chords: [
      { deg: [0, 2, 4, 6], oct: [0, 0, 0, 0], rollMs: 148, vel: 0.50 },
      { deg: [5, 0, 2, 4], oct: [-1, 0, 0, 0], rollMs: 162, vel: 0.46 },
      { deg: [3, 5, 0, 2], oct: [0, 0, 1, 1], rollMs: 154, vel: 0.48 },
      { deg: [4, 6, 1, 3], oct: [0, 0, 1, 1], rollMs: 144, vel: 0.45 },
    ],
  },
  {
    id: 'mixing_colours',
    name: 'Felt & Mist',
    label: 'Felt & Mist (G Dorian)',
    keyLabel: 'G Dorian',
    tonic: 'G',
    rootMidi: 55,
    intervals: [0, 2, 3, 5, 7, 9, 10],
    degreeWeights: [0.95, 0.70, 0.80, 0.35, 0.85, 0.65, 0.45],
    markovGravity: [1.48, 1.0, 1.55, 1.0, 1.42, 1.38, 1.0],
    rotationSpeed: 1.00,
    pianoCutoffHz: 860,
    pianoShelfDb: -12.5,
    globalAttackSec: 2.8,
    phraseCycleSec: 13.8,
    phraseGateProb: 0.84,
    phraseLengthLfo: { enabled: false },
    droneMode: 'Breathing wash',
    washInstrument: 'pad',
    washFade: 4.2,
    washSustain: 6.8,
    washGap: 2.5,
    chordInstrument: 'pad',
    chordRollMs: 72,
    chordVel: 0.44,
    voices: {
      pad:   { cycle: 18.9, prob: 0.55, gain: 0.60, attack: 2.8, release: 7.2, octave: 0, pan: 0.34, muted: false },
      bell:  { cycle: 21.3, prob: 0.28, gain: 0.50, attack: 0.04, release: 4.2, octave: 1, pan: 0.55, muted: false },
      choir: { cycle: 27.1, prob: 0.38, gain: 0.56, attack: 3.4, release: 8.0, octave: 1, pan: 0.46, muted: false },
      tape:  { cycle: 17.7, prob: 0.56, gain: 0.58, attack: 2.2, release: 5.2, octave: 0, pan: 0.38, muted: false },
      drone: { cycle: 31.3, prob: 0.0,  gain: 0.62, attack: 3.6, release: 9.0, octave: -1, pan: 0.14, muted: true },
    },
    phrases: [
      // Winter Bell Cell & Mist & Memory Sequence from eno_soundscape/js/phrase-patterns.js
      [
        { deg: [0], oct: [1], delayMs: 0, vel: 0.40, rollMs: 68 },
        { deg: [2], oct: [1], delayMs: 520, vel: 0.34, rollMs: 68 },
        { deg: [1], oct: [1], delayMs: 1480, vel: 0.31, rollMs: 72 },
        { deg: [4], oct: [0], delayMs: 2080, vel: 0.38, rollMs: 76 },
      ],
      [
        { deg: [0], oct: [1], delayMs: 0, vel: 0.40, rollMs: 68 },
        { deg: [1], oct: [1], delayMs: 540, vel: 0.33, rollMs: 68 },
        { deg: [2], oct: [1], delayMs: 1120, vel: 0.36, rollMs: 72 },
        { deg: [4], oct: [1], delayMs: 1960, vel: 0.39, rollMs: 75 },
        { deg: [3], oct: [1], delayMs: 2580, vel: 0.34, rollMs: 70 },
        { deg: [1], oct: [1], delayMs: 3200, vel: 0.31, rollMs: 68 },
        { deg: [0], oct: [1], delayMs: 4100, vel: 0.38, rollMs: 78 },
      ],
    ],
    chords: [
      { deg: [0, 2, 4, 6], oct: [0, 0, 0, 0], rollMs: 72, vel: 0.44 },
      { deg: [3, 5, 0, 2], oct: [0, 0, 1, 1], rollMs: 78, vel: 0.40 },
      { deg: [6, 1, 3, 5], oct: [-1, 0, 0, 0], rollMs: 68, vel: 0.38 },
      { deg: [4, 6, 1, 3], oct: [0, 0, 1, 1], rollMs: 84, vel: 0.42 },
    ],
  },
  {
    id: 'discreet_loops',
    name: 'Moss & Reel',
    label: 'Moss & Reel (D Aeolian)',
    keyLabel: 'D Aeolian',
    tonic: 'D',
    rootMidi: 50,
    intervals: [0, 2, 3, 5, 7, 8, 10],
    degreeWeights: [0.90, 0.50, 0.75, 0.45, 0.80, 0.60, 0.40],
    markovGravity: [1.58, 1.0, 1.48, 1.0, 1.52, 1.38, 1.32],
    rotationSpeed: 1.05,
    pianoCutoffHz: 910,
    pianoShelfDb: -11.5,
    globalAttackSec: 2.4,
    phraseCycleSec: 14.6,
    phraseGateProb: 0.86,
    phraseLengthLfo: { enabled: false },
    droneMode: 'Breathing wash',
    washInstrument: 'pad',
    washFade: 4.4,
    washSustain: 7.0,
    washGap: 3.0,
    chordInstrument: 'tape',
    chordRollMs: 136,
    chordVel: 0.48,
    // In eno_soundscape/js/presets.js, discreet_loops has activeVoiceCount: 4 (bell & choir muted)
    voices: {
      pad:   { cycle: 18.7, prob: 0.60, gain: 0.62, attack: 2.8, release: 7.9, octave: 0, pan: 0.36, muted: false },
      bell:  { cycle: 19.3, prob: 0.0,  gain: 0.50, attack: 0.05, release: 4.0, octave: 1, pan: 0.60, muted: true },
      choir: { cycle: 25.9, prob: 0.0,  gain: 0.55, attack: 3.4, release: 7.5, octave: 1, pan: 0.48, muted: true },
      tape:  { cycle: 14.9, prob: 0.74, gain: 0.65, attack: 2.2, release: 8.0, octave: 0, pan: 0.42, muted: false },
      drone: { cycle: 26.3, prob: 0.0,  gain: 0.65, attack: 3.4, release: 9.5, octave: -1, pan: 0.16, muted: true },
    },
    phrases: [
      // Discreet Tape Fragment & Earth & Overgrowth Arc from eno_soundscape/js/phrase-patterns.js
      [
        { deg: [0], oct: [1], delayMs: 0, vel: 0.44, rollMs: 125 },
        { deg: [2], oct: [1], delayMs: 610, vel: 0.36, rollMs: 125 },
        { deg: [4], oct: [1], delayMs: 1220, vel: 0.40, rollMs: 135 },
        { deg: [1], oct: [1], delayMs: 2440, vel: 0.34, rollMs: 145 },
      ],
      [
        { deg: [0], oct: [1], delayMs: 0, vel: 0.44, rollMs: 125 },
        { deg: [1], oct: [1], delayMs: 610, vel: 0.35, rollMs: 125 },
        { deg: [2], oct: [1], delayMs: 1220, vel: 0.39, rollMs: 130 },
        { deg: [4], oct: [1], delayMs: 1830, vel: 0.42, rollMs: 135 },
        { deg: [3], oct: [1], delayMs: 3050, vel: 0.36, rollMs: 130 },
        { deg: [1], oct: [1], delayMs: 3660, vel: 0.34, rollMs: 125 },
        { deg: [0], oct: [1], delayMs: 4880, vel: 0.41, rollMs: 140 },
      ],
    ],
    chords: [
      { deg: [0, 2, 4, 6], oct: [0, 0, 0, 0], rollMs: 128, vel: 0.48 },
      { deg: [5, 0, 2, 4], oct: [-1, 0, 0, 0], rollMs: 142, vel: 0.44 },
      { deg: [2, 4, 6, 1], oct: [0, 0, 0, 1], rollMs: 132, vel: 0.42 },
      { deg: [6, 1, 3, 5], oct: [-1, 0, 0, 0], rollMs: 148, vel: 0.46 },
    ],
  },
];

export const TRACKS = ENSO_TRACKS;

export const ENSO_TRACK_NAMES = ENSO_TRACKS.map((t) => {
  if (!t.scale) t.scale = t.intervals;
  return t.label;
});

function midiToHz(midi) {
  return 440 * Math.pow(2, (midi - 69) / 12);
}

const SAMPLE_SOURCE_RANK = { sync: 1, offline: 2, mp3: 3, ogg: 4 };

export function createEnsoSound({ onTrackChange, onStateChange } = {}) {
  let ctx = null;
  let masterGain = null;
  let dryBus = null;
  let pianoFeltFilter = null;
  let pianoSilkShelf = null;
  let bgWarmFilter = null;
  let fdnSend = null;
  let fdnNodes = [];
  let enabled = false;
  let volume = 0.75;
  let washLevel = 0.33;
  let autoRotate = true;
  let reverbMode = ENSO_REVERB_MODES[0];
  let trackIndex = 0;
  let droneMode = ENSO_TRACKS[trackIndex].droneMode || 'Breathing wash';
  let washInstrument = ENSO_TRACKS[trackIndex].washInstrument || 'choir';
  let washFade = ENSO_TRACKS[trackIndex].washFade ?? 4.2;
  let washSustain = ENSO_TRACKS[trackIndex].washSustain ?? 7.0;
  let washGap = ENSO_TRACKS[trackIndex].washGap ?? 3.0;
  let trackElapsedSec = 0;
  let totalListenedSec = 0;
  let phraseTimerSec = 15.0;
  let phraseCycleCount = 0;
  let chordStepIdx = 0;
  let currentChordDegrees = [0, 2, 4];
  let noteHistory = [];
  let bgVoiceTimers = [];
  let pianoLevel = 0.85;
  let notesTriggered = 0;
  let pianoNotesTriggered = 0;
  let lastPianoMidi = 60;
  let lastPianoSampleMidi = 60;
  let lastPianoPlaybackRate = 1.0;
  let lastPianoPeakGain = 0.0;
  let pianoNoteLog = [];
  let lastAppliedWashTarget = -1;
  let oggSamplesLoaded = 0;
  let offlineFallbackReady = false;
  let sampleLoadStarted = false;
  let preferredFormat = 'ogg';

  // Persistent Harmonic Synth Drone / Wash Engine (0% sample decay, 0% ConvolverNode CPU)
  const VIBRATO_RATE_HZ = 5.2;
  const VIBRATO_DEPTH_CENTS = 3.0;
  let washMasterGain = null;
  let washEnvGain = null;
  let washAnalyser = null;
  let washAnalyserBuf = null;
  let pianoAnalyser = null;
  let pianoAnalyserBuf = null;
  let washFormantFilter = null;
  let washSilkFilter = null;
  let washWarmShelf = null;
  let washFdnTap = null;
  let droneBank = [];
  let vibratoLfo = null;
  let vibratoGain = null;
  let washPhase = 'fadeIn';
  let washPhaseElapsed = 0;
  let washStartFloor = 0.0;
  let washEnvLevel = 0.0;
  let washTimerSec = 6.2;
  let currentWashCycleSec = 18.4;
  let washesTriggered = 0;
  let transitionActive = false;
  let transitionStartLevel = 0.0;
  let transitionOutSec = 4.5;
  let transitionGapSec = 1.5;
  let pendingRetune = false;
  let lastWashChordMidis = [];
  let lastWashTrackId = ENSO_TRACKS[0].id;
  let lastWashInstrument = ENSO_TRACKS[0].washInstrument || 'choir';
  let lastWashVoices = [];
  let activeWashes = [];

  function applyTrackDroneDefaults(track) {
    if (!track) return;
    droneMode = track.droneMode || 'Breathing wash';
    washInstrument = track.washInstrument || 'choir';
    washFade = Number.isFinite(track.washFade) ? track.washFade : 4.2;
    washSustain = Number.isFinite(track.washSustain) ? track.washSustain : 7.0;
    washGap = Number.isFinite(track.washGap) ? track.washGap : (droneMode === 'Continuous drone' ? 0 : 3.0);
  }

  // Map<voice, Array<{ midi, buffer, source: 'ogg'|'mp3'|'offline' }>>
  const sampleBank = new Map();
  const scheduledTimers = new Set();

  // Staggered initial phases across the 5 non-piano voices (piano is driven by the Enso Phrase Engine)
  const voicePhases = {
    pad: 0.45,
    bell: 0.12,
    choir: 0.68,
    tape: 0.22,
    drone: 0.55,
  };

  function detectAudioFormat() {
    if (typeof document === 'undefined') return 'ogg';
    try {
      const a = document.createElement('audio');
      if (a && typeof a.canPlayType === 'function') {
        const canOgg = a.canPlayType('audio/ogg; codecs="vorbis"');
        if (canOgg === 'probably' || canOgg === 'maybe') return 'ogg';
      }
    } catch {}
    return 'mp3';
  }

  function registerSample(voice, midi, buffer, sourceTag) {
    if (!sampleBank.has(voice)) sampleBank.set(voice, []);
    const list = sampleBank.get(voice);
    const existingIdx = list.findIndex((x) => x.midi === midi);
    const entry = { midi, buffer, source: sourceTag };
    if (existingIdx >= 0) {
      const prevRank = SAMPLE_SOURCE_RANK[list[existingIdx].source] || 0;
      const nextRank = SAMPLE_SOURCE_RANK[sourceTag] || 0;
      if (nextRank < prevRank) return;
      list[existingIdx] = entry;
    } else {
      list.push(entry);
      list.sort((a, b) => a.midi - b.midi);
    }
  }

  // Synchronous 0ms warm stereo bootstrap buffers so the very first wash & phrase play immediately
  // even before OfflineAudioContext or network .ogg decoding completes.
  function buildSyncBootstrapSamples(audioCtx) {
    if (!audioCtx || sampleBank.size > 0) return;
    const rate = 22050;
    const dur = 3.2;
    const frames = Math.floor(rate * dur);
    const bootstrapZones = [
      { voice: 'piano', midi: 60 },
      { voice: 'pad',   midi: 55 },
      { voice: 'choir', midi: 60 },
      { voice: 'drone', midi: 43 },
      { voice: 'bell',  midi: 67 },
      { voice: 'tape',  midi: 55 },
    ];
    for (const z of bootstrapZones) {
      try {
        const buf = audioCtx.createBuffer(2, frames, rate);
        const chL = buf.getChannelData(0);
        const chR = buf.getChannelData(1);
        const f0 = midiToHz(z.midi);
        const isPerc = z.voice === 'piano' || z.voice === 'bell';
        for (let i = 0; i < frames; i++) {
          const t = i / rate;
          const env = isPerc
            ? Math.min(1, t / 0.01) * Math.exp(-1.1 * t)
            : Math.min(1, t / 0.35) * Math.min(1, (dur - t) / 0.35);
          const sL = 0.32 * Math.sin(2 * Math.PI * f0 * t) + 0.10 * Math.sin(2 * Math.PI * (f0 * 2.0) * t + 0.3);
          const sR = 0.32 * Math.sin(2 * Math.PI * (f0 * 1.0012) * t + 0.6) + 0.10 * Math.sin(2 * Math.PI * (f0 * 1.9988) * t + 1.1);
          chL[i] = sL * env;
          chR[i] = sR * env;
        }
        registerSample(z.voice, z.midi, buf, 'sync');
      } catch {}
    }
  }

  // Non-ringing, critically damped (Q = 0.5) 4-tap Feedback Delay Network for optional FDN modes
  function buildLiteFdnReverb(audioCtx, destNode) {
    const input = audioCtx.createGain();
    input.gain.value = 0.0;

    const preLp = audioCtx.createBiquadFilter();
    preLp.type = 'lowpass';
    preLp.frequency.value = 1350;
    preLp.Q.value = 0.5;

    const ap1 = audioCtx.createBiquadFilter();
    ap1.type = 'allpass';
    ap1.frequency.value = 540;
    ap1.Q.value = 0.5;

    const ap2 = audioCtx.createBiquadFilter();
    ap2.type = 'allpass';
    ap2.frequency.value = 1180;
    ap2.Q.value = 0.5;

    input.connect(preLp);
    preLp.connect(ap1);
    ap1.connect(ap2);

    const delayTimes = [0.079, 0.113, 0.149, 0.193];
    const merger = audioCtx.createChannelMerger(2);
    const outGain = audioCtx.createGain();
    outGain.gain.value = 0.30;

    const nodes = [input, preLp, ap1, ap2, merger, outGain];
    const delays = [];

    for (let i = 0; i < delayTimes.length; i++) {
      const d = audioCtx.createDelay(0.35);
      d.delayTime.value = delayTimes[i];
      const damp = audioCtx.createBiquadFilter();
      damp.type = 'lowpass';
      damp.frequency.value = 1200;
      damp.Q.value = 0.5;
      const fb = audioCtx.createGain();
      fb.gain.value = 0.42;

      ap2.connect(d);
      d.connect(damp);
      damp.connect(fb);
      d.connect(merger, 0, i % 2);

      delays.push({ d, damp, fb });
      nodes.push(d, damp, fb);
    }

    for (let i = 0; i < delays.length; i++) {
      const next = delays[(i + 1) % delays.length];
      delays[i].fb.connect(next.d);
    }

    merger.connect(outGain);
    outGain.connect(destNode);
    fdnNodes = nodes;
    return input;
  }

  function syncReverbMode() {
    if (!fdnSend || !ctx) return;
    const sendVal =
      reverbMode === 'Lite FDN Live' ? 0.36 :
      reverbMode === 'Hybrid (OGG + Lite FDN)' ? 0.12 :
      0.0;
    fdnSend.gain.setTargetAtTime(sendVal, ctx.currentTime, 0.05);
  }

  function syncTrackPianoFilter(track) {
    if (!ctx || !pianoFeltFilter || !pianoSilkShelf) return;
    const now = ctx.currentTime;
    pianoFeltFilter.frequency.setTargetAtTime(track.pianoCutoffHz || 1050, now, 0.12);
    pianoSilkShelf.gain.setTargetAtTime(track.pianoShelfDb ?? -8.0, now, 0.12);
  }

  // Instant zero-network fallback: renders smooth, dark-damped (Q = 0.5, zero ringing) root buffers
  // locally via OfflineAudioContext + ConvolverNode in ~25ms while 320kbps .ogg files stream in.
  async function buildOfflineStampedFallback() {
    const OfflineCtx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
    if (!OfflineCtx) return;
    const quickZones = [
      { voice: 'piano', midi: 48, dur: 4.5 },
      { voice: 'piano', midi: 60, dur: 4.2 },
      { voice: 'piano', midi: 72, dur: 4.0 },
      { voice: 'pad',   midi: 55, dur: 5.2 },
      { voice: 'bell',  midi: 67, dur: 4.2 },
      { voice: 'choir', midi: 60, dur: 5.2 },
      { voice: 'tape',  midi: 55, dur: 4.6 },
      { voice: 'drone', midi: 43, dur: 5.4 },
    ];

    const rate = 24000;
    await Promise.all(quickZones.map(async (z) => {
      try {
        const frames = Math.floor(rate * z.dur);
        const octx = new OfflineCtx(2, frames, rate);
        const freq = 440 * Math.pow(2, (z.midi - 69) / 12);

        const irDur = Math.min(z.dur - 0.3, 3.4);
        const irLen = Math.floor(rate * irDur);
        const irBuf = octx.createBuffer(2, irLen, rate);
        const irL = irBuf.getChannelData(0);
        const irR = irBuf.getChannelData(1);
        let s = (z.midi * 2654435761) >>> 0;
        const rand = () => { s = (s * 1664525 + 1013904223) >>> 0; return (s / 4294967296) * 2 - 1; };
        const decay = 3.8 / irDur;
        let lpL = 0, lpR = 0;
        for (let i = 0; i < irLen; i++) {
          const t = i / rate;
          const env = 0.5 * (1 - Math.cos(Math.PI * Math.min(1, t / 0.025))) * Math.exp(-decay * t);
          lpL += 0.16 * (rand() * env - lpL);
          lpR += 0.16 * (rand() * env - lpR);
          irL[i] = lpL;
          irR[i] = lpR;
        }

        const conv = octx.createConvolver();
        conv.buffer = irBuf;
        const damp = octx.createBiquadFilter();
        damp.type = 'lowpass';
        damp.frequency.value = z.voice === 'piano' ? 1300 : 1700;
        damp.Q.value = 0.5;
        const wet = octx.createGain();
        wet.gain.value = z.voice === 'piano' ? 0.30 : 0.45;
        damp.connect(conv);
        conv.connect(wet);
        wet.connect(octx.destination);

        const env = octx.createGain();
        const isPerc = z.voice === 'piano' || z.voice === 'bell';
        env.gain.setValueAtTime(0.0001, 0);
        if (isPerc) {
          env.gain.linearRampToValueAtTime(0.60, 0.010);
          env.gain.exponentialRampToValueAtTime(0.0001, z.dur - 0.15);
        } else {
          // Flat sustained plateau for pad/choir/drone/tape so looped washes never decay early
          env.gain.linearRampToValueAtTime(0.56, 0.45);
          env.gain.setValueAtTime(0.56, z.dur - 0.40);
          env.gain.linearRampToValueAtTime(0.0001, z.dur - 0.04);
        }
        env.connect(octx.destination);
        env.connect(damp);

        const o1 = octx.createOscillator();
        o1.type = z.voice === 'pad' ? 'triangle' : 'sine';
        o1.frequency.value = freq;
        const o2 = octx.createOscillator();
        o2.type = 'sine';
        o2.frequency.value = freq * 2.0;
        const g2 = octx.createGain();
        g2.gain.value = z.voice === 'piano' ? 0.16 : 0.18;
        const lp = octx.createBiquadFilter();
        lp.type = 'lowpass';
        lp.frequency.value = z.voice === 'piano' ? 950 : Math.min(1800, freq * 2.5);
        lp.Q.value = 0.5;
        o1.connect(lp);
        o2.connect(g2);
        g2.connect(lp);
        lp.connect(env);
        o1.start(0);
        o2.start(0);
        o1.stop(z.dur - 0.1);
        o2.stop(z.dur - 0.1);

        const rendered = await octx.startRendering();
        registerSample(z.voice, z.midi, rendered, 'offline');
      } catch {}
    }));
    offlineFallbackReady = true;
  }

  // Stream & decode the 320 kbps reverb-stamped .ogg samples (with 320 kbps .mp3 fallback)
  async function loadServerSamples() {
    if (sampleLoadStarted || !ctx) return;
    sampleLoadStarted = true;
    preferredFormat = detectAudioFormat();

    const fetchZone = async (zone) => {
      const urls = preferredFormat === 'ogg'
        ? [`samples/${zone.id}.ogg`, `samples/${zone.id}.mp3`]
        : [`samples/${zone.id}.mp3`, `samples/${zone.id}.ogg`];
      for (const url of urls) {
        try {
          const resp = await fetch(url);
          if (!resp.ok) continue;
          const ab = await resp.arrayBuffer();
          const decoded = await ctx.decodeAudioData(ab);
          if (decoded) {
            const fmt = url.endsWith('.ogg') ? 'ogg' : 'mp3';
            registerSample(zone.voice, zone.midi, decoded, fmt);
            oggSamplesLoaded++;
            return true;
          }
        } catch {}
      }
      return false;
    };

    const priority = SAMPLE_ZONES.filter((z) => z.midi >= 48 && z.midi <= 72);
    const outer = SAMPLE_ZONES.filter((z) => z.midi < 48 || z.midi > 72);
    const ordered = [...priority, ...outer];
    const BATCH_SIZE = 4;
    for (let i = 0; i < ordered.length; i += BATCH_SIZE) {
      await Promise.all(ordered.slice(i, i + BATCH_SIZE).map(fetchZone));
    }
  }

  function scheduleSamplePreloadAfterFirstFrame() {
    const startPreload = () => {
      buildOfflineStampedFallback();
      loadServerSamples();
    };
    if (typeof window !== 'undefined' && typeof window.requestAnimationFrame === 'function') {
      window.requestAnimationFrame(() => {
        setTimeout(startPreload, 150);
      });
    } else {
      setTimeout(startPreload, 150);
    }
  }

  function ensureContext() {
    if (ctx) return ctx;
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return null;
    try {
      ctx = new AudioCtx({ latencyHint: 'playback' });
    } catch {
      ctx = new AudioCtx();
    }

    masterGain = ctx.createGain();
    masterGain.gain.value = enabled ? volume : 0;
    masterGain.connect(ctx.destination);

    dryBus = ctx.createGain();
    dryBus.gain.value = 1.0;
    dryBus.connect(masterGain);

    // Live per-track 3-stage Felt Piano bus filters (Q = 0.5, zero resonance peak)
    // Ensures pitch-shifted piano samples stay warm, soft, and completely ring-free
    const track = ENSO_TRACKS[trackIndex];
    pianoFeltFilter = ctx.createBiquadFilter();
    pianoFeltFilter.type = 'lowpass';
    pianoFeltFilter.frequency.value = track.pianoCutoffHz || 1050;
    pianoFeltFilter.Q.value = 0.5;

    pianoSilkShelf = ctx.createBiquadFilter();
    pianoSilkShelf.type = 'highshelf';
    pianoSilkShelf.frequency.value = 1350;
    pianoSilkShelf.Q.value = 0.5;
    pianoSilkShelf.gain.value = track.pianoShelfDb ?? -8.0;

    pianoFeltFilter.connect(pianoSilkShelf);
    pianoSilkShelf.connect(dryBus);
    if (ctx.createAnalyser) {
      pianoAnalyser = ctx.createAnalyser();
      pianoAnalyser.fftSize = 2048;
      pianoAnalyserBuf = new Float32Array(pianoAnalyser.fftSize);
      pianoSilkShelf.connect(pianoAnalyser);
    }

    // Warm lowpass bus for background pads/choir/tape/drone/bell (Q = 0.5)
    bgWarmFilter = ctx.createBiquadFilter();
    bgWarmFilter.type = 'lowpass';
    bgWarmFilter.frequency.value = 1850;
    bgWarmFilter.Q.value = 0.5;
    bgWarmFilter.connect(dryBus);

    fdnSend = buildLiteFdnReverb(ctx, masterGain);
    syncReverbMode();

    // Dedicated Persistent Harmonic Synth Drone / Wash bus (0% sample decay, warm 2-stage Q = 0.5 silk filters)
    washMasterGain = ctx.createGain();
    washMasterGain.gain.value = washLevel;
    washMasterGain.connect(dryBus);

    washFdnTap = ctx.createGain();
    washFdnTap.gain.value = 0.42;
    washMasterGain.connect(washFdnTap);
    if (fdnSend) washFdnTap.connect(fdnSend);

    washEnvGain = ctx.createGain();
    washEnvGain.gain.value = 0.0;
    washEnvGain.connect(washMasterGain);

    if (ctx.createAnalyser) {
      washAnalyser = ctx.createAnalyser();
      washAnalyser.fftSize = 2048;
      washAnalyserBuf = new Float32Array(washAnalyser.fftSize);
      washEnvGain.connect(washAnalyser);
    }

    washFormantFilter = ctx.createBiquadFilter();
    washFormantFilter.type = 'peaking';
    washFormantFilter.frequency.value = 260;
    washFormantFilter.Q.value = 0.55;
    washFormantFilter.gain.value = 2.0;

    washSilkFilter = ctx.createBiquadFilter();
    washSilkFilter.type = 'lowpass';
    washSilkFilter.frequency.value = 540;
    washSilkFilter.Q.value = 0.5;

    washWarmShelf = ctx.createBiquadFilter();
    washWarmShelf.type = 'highshelf';
    washWarmShelf.frequency.value = 1150;
    washWarmShelf.Q.value = 0.5;
    washWarmShelf.gain.value = -8.5;

    washFormantFilter.connect(washSilkFilter);
    washSilkFilter.connect(washWarmShelf);
    washWarmShelf.connect(washEnvGain);

    initPersistentDroneBank(ctx, track);

    buildSyncBootstrapSamples(ctx);
    scheduleSamplePreloadAfterFirstFrame();
    return ctx;
  }

  function computeWashVoiceSpecs(track, instName) {
    const rawInst = instName || track.washInstrument || 'choir';
    const washVoice = (rawInst === 'piano' || rawInst === 'bell') ? 'pad' : rawInst;
    const scale = track.scale || track.intervals || [0, 2, 4, 5, 7, 9, 11];
    const rootMidi = track.rootMidi || 48;
    const fifthMidi = rootMidi + (scale[4] ?? 7);
    const thirdMidi = rootMidi + (scale[2] ?? 4) + 12;
    const octaveMidi = rootMidi + 12;
    const ninthMidi = rootMidi + (scale[1] ?? 2) + 12;

    // 6 orthogonal, pitch-locked harmonic partials per track (distinct pitches -> zero phase beating, zero pitch drift)
    let specs;
    let filterCfg;
    if (washVoice === 'drone') {
      specs = [
        { voice: washVoice, type: 'sine',     midi: rootMidi - 12,   gain: 0.135, pan: -0.06 },
        { voice: washVoice, type: 'triangle', midi: rootMidi,        gain: 0.112, pan: -0.22 },
        { voice: washVoice, type: 'sine',     midi: fifthMidi - 12,  gain: 0.092, pan: +0.22 },
        { voice: washVoice, type: 'sine',     midi: octaveMidi,      gain: 0.056, pan: +0.14 },
        { voice: washVoice, type: 'triangle', midi: fifthMidi,       gain: 0.048, pan: -0.26 },
        { voice: washVoice, type: 'sine',     midi: ninthMidi - 12,  gain: 0.046, pan: +0.24 },
      ];
      filterCfg = { cutoffHz: 460, q: 0.5, formantHz: 190, formantGainDb: 2.2, shelfDb: -9.5 };
    } else if (washVoice === 'choir') {
      specs = [
        { voice: washVoice, type: 'sine',     midi: rootMidi,        gain: 0.105, pan: -0.20 },
        { voice: washVoice, type: 'triangle', midi: fifthMidi,       gain: 0.092, pan: +0.22 },
        { voice: washVoice, type: 'triangle', midi: octaveMidi,      gain: 0.084, pan: -0.28 },
        { voice: washVoice, type: 'sine',     midi: ninthMidi,       gain: 0.076, pan: +0.28 },
        { voice: washVoice, type: 'sine',     midi: thirdMidi,       gain: 0.062, pan: -0.14 },
        { voice: washVoice, type: 'sine',     midi: fifthMidi + 12,  gain: 0.048, pan: +0.16 },
      ];
      filterCfg = { cutoffHz: 860, q: 0.55, formantHz: 560, formantGainDb: 3.0, shelfDb: -7.0 };
    } else if (washVoice === 'tape') {
      specs = [
        { voice: washVoice, type: 'sine',     midi: rootMidi - 12,   gain: 0.075, pan: -0.08 },
        { voice: washVoice, type: 'triangle', midi: rootMidi,        gain: 0.110, pan: -0.24 },
        { voice: washVoice, type: 'triangle', midi: fifthMidi - 12,  gain: 0.082, pan: +0.24 },
        { voice: washVoice, type: 'sine',     midi: fifthMidi,       gain: 0.076, pan: -0.20 },
        { voice: washVoice, type: 'triangle', midi: octaveMidi,      gain: 0.064, pan: +0.26 },
        { voice: washVoice, type: 'sine',     midi: thirdMidi,       gain: 0.058, pan: +0.16 },
      ];
      filterCfg = { cutoffHz: 620, q: 0.5, formantHz: 290, formantGainDb: 2.2, shelfDb: -8.5 };
    } else {
      // 'pad' (warm CS-80 ribbon pad)
      specs = [
        { voice: washVoice, type: 'sine',     midi: rootMidi - 12,   gain: 0.085, pan: -0.08 },
        { voice: washVoice, type: 'triangle', midi: rootMidi,        gain: 0.112, pan: -0.26 },
        { voice: washVoice, type: 'triangle', midi: fifthMidi,       gain: 0.088, pan: +0.26 },
        { voice: washVoice, type: 'sine',     midi: octaveMidi,      gain: 0.072, pan: -0.18 },
        { voice: washVoice, type: 'sine',     midi: thirdMidi,       gain: 0.064, pan: +0.22 },
        { voice: washVoice, type: 'sine',     midi: fifthMidi + 12,  gain: 0.044, pan: -0.14 },
      ];
      filterCfg = { cutoffHz: 720, q: 0.5, formantHz: 340, formantGainDb: 1.8, shelfDb: -8.0 };
    }
    return { washVoice, specs, filterCfg };
  }

  function updateWashTargetMetadata(track = ENSO_TRACKS[trackIndex], instName = washInstrument) {
    const { washVoice, specs, filterCfg } = computeWashVoiceSpecs(track, instName);
    lastWashChordMidis = specs.map((s) => s.midi);
    lastWashTrackId = track.id;
    lastWashInstrument = washVoice;
    lastWashVoices = specs.map(() => washVoice);
    return { washVoice, specs, filterCfg };
  }

  function initPersistentDroneBank(audioCtx, track) {
    if (droneBank.length > 0 || !audioCtx || !washFormantFilter) return;
    const now = audioCtx.currentTime;
    const { specs, filterCfg } = updateWashTargetMetadata(track, washInstrument);

    washSilkFilter.frequency.setValueAtTime(filterCfg.cutoffHz, now);
    washSilkFilter.Q.setValueAtTime(filterCfg.q, now);
    washFormantFilter.frequency.setValueAtTime(filterCfg.formantHz, now);
    washFormantFilter.gain.setValueAtTime(filterCfg.formantGainDb, now);
    washWarmShelf.gain.setValueAtTime(filterCfg.shelfDb, now);

    // Subtle vibrato-rate LFO (5.2 Hz, ±3.0 cents) for natural shimmer without slow rising/falling pitch drift
    vibratoLfo = audioCtx.createOscillator();
    vibratoLfo.type = 'sine';
    vibratoLfo.frequency.setValueAtTime(VIBRATO_RATE_HZ, now);
    vibratoGain = audioCtx.createGain();
    vibratoGain.gain.setValueAtTime(VIBRATO_DEPTH_CENTS, now);
    vibratoLfo.connect(vibratoGain);
    vibratoLfo.start(now);

    for (let i = 0; i < specs.length; i++) {
      const sp = specs[i];
      const osc = audioCtx.createOscillator();
      osc.type = sp.type;
      const hz = midiToHz(sp.midi);
      osc.frequency.setValueAtTime(hz, now);
      osc.frequency.value = hz;
      if (vibratoGain) vibratoGain.connect(osc.detune);

      const vGain = audioCtx.createGain();
      vGain.gain.setValueAtTime(sp.gain * 1.45, now);

      let panner = null;
      if (audioCtx.createStereoPanner) {
        panner = audioCtx.createStereoPanner();
        panner.pan.setValueAtTime(sp.pan, now);
        osc.connect(vGain);
        vGain.connect(panner);
        panner.connect(washFormantFilter);
      } else {
        osc.connect(vGain);
        vGain.connect(washFormantFilter);
      }

      osc.start(now);
      droneBank.push({ osc, vGain, panner, midi: sp.midi, gain: sp.gain, type: sp.type });
    }
  }

  // Strictly snaps oscillator base frequencies via setValueAtTime ONLY when washEnvLevel === 0 (silent gap).
  // Never calls setTargetAtTime or linearRampToValueAtTime on oscillator frequency!
  function retuneDroneBank(track = ENSO_TRACKS[trackIndex], { forceAtZero = false } = {}) {
    const { washVoice, specs, filterCfg } = updateWashTargetMetadata(track, washInstrument);

    if (!ctx || !droneBank.length) return { washVoice, specs };
    if (enabled && washEnvLevel > 0.0001 && !forceAtZero) {
      pendingRetune = true;
      return { washVoice, specs };
    }

    pendingRetune = false;
    const now = ctx.currentTime;

    if (washSilkFilter && washFormantFilter && washWarmShelf) {
      washSilkFilter.frequency.cancelScheduledValues(now);
      washSilkFilter.frequency.setValueAtTime(filterCfg.cutoffHz, now);
      washSilkFilter.Q.setValueAtTime(filterCfg.q, now);
      washFormantFilter.frequency.cancelScheduledValues(now);
      washFormantFilter.frequency.setValueAtTime(filterCfg.formantHz, now);
      washFormantFilter.gain.cancelScheduledValues(now);
      washFormantFilter.gain.setValueAtTime(filterCfg.formantGainDb, now);
      washWarmShelf.gain.cancelScheduledValues(now);
      washWarmShelf.gain.setValueAtTime(filterCfg.shelfDb, now);
    }

    for (let i = 0; i < droneBank.length && i < specs.length; i++) {
      const node = droneBank[i];
      const sp = specs[i];
      const targetHz = midiToHz(sp.midi);
      node.osc.type = sp.type;
      node.osc.frequency.cancelScheduledValues(now);
      node.osc.frequency.setValueAtTime(targetHz, now);
      node.osc.frequency.value = targetHz;
      node.vGain.gain.cancelScheduledValues(now);
      node.vGain.gain.setValueAtTime(sp.gain * 1.45, now);
      if (node.panner) {
        node.panner.pan.cancelScheduledValues(now);
        node.panner.pan.setValueAtTime(sp.pan, now);
      }
      node.midi = sp.midi;
      node.gain = sp.gain;
      node.type = sp.type;
    }
    return { washVoice, specs };
  }

  function applyWashEnvelopeGain(level, timeConstant = 0.08) {
    const clamped = Math.max(0, Math.min(1, Number(level) || 0));
    washEnvLevel = clamped <= 0.0001 ? 0 : clamped;
    if (ctx && washEnvGain) {
      if (Math.abs(washEnvLevel - lastAppliedWashTarget) > 1e-4) {
        lastAppliedWashTarget = washEnvLevel;
        const now = ctx.currentTime;
        washEnvGain.gain.setTargetAtTime(washEnvLevel, now, washEnvLevel === 0 ? 0.045 : timeConstant);
      }
    }
  }

  function refreshActiveWashesRecord(track = ENSO_TRACKS[trackIndex]) {
    if (!enabled || washPhase === 'gap' || washPhase === 'transitionGap' || washEnvLevel <= 0.0005) {
      activeWashes = [];
    } else {
      const fadeSec = Math.max(1.5, Math.min(10.0, Number(washFade) || 4.2));
      const sustainSec = Math.max(3.0, Math.min(12.0, Number(washSustain) || 7.0));
      const totalDur = droneMode === 'Continuous drone' ? fadeSec * 3.5 : (fadeSec * 2 + sustainSec);
      activeWashes = [{
        id: Math.max(1, washesTriggered),
        trackId: track.id,
        instrument: lastWashInstrument,
        mode: droneMode,
        phase: washPhase,
        envLevel: Number(washEnvLevel.toFixed(4)),
        remainingSec: washPhase === 'continuous' ? 999 : Math.max(0.1, totalDur - washPhaseElapsed),
        totalDur,
        chordMidis: lastWashChordMidis.slice(),
      }];
    }
  }

  function triggerHeavenlyWash(track = ENSO_TRACKS[trackIndex], { initial = false, crossfadeInSec = washFade } = {}) {
    if (!enabled) return false;
    const audioCtx = ensureContext();
    if (!audioCtx || !washSilkFilter) return false;
    if (audioCtx.state === 'suspended') {
      audioCtx.resume().catch(() => {});
    }

    const fadeSec = Math.max(1.5, Math.min(10.0, Number(crossfadeInSec || washFade) || 4.2));
    const sustainSec = Math.max(3.0, Math.min(12.0, Number(washSustain) || 7.0));
    const gapSec = Math.max(0, Math.min(8.0, Number(washGap) || 0));
    const isContinuous = (droneMode === 'Continuous drone') && (gapSec <= 0.05);

    if (!initial && washEnvLevel > 0.0005) {
      // Smooth ~10s track transition: fade out current wash at locked pitch -> silent gap -> retune at 0 -> fade in
      updateWashTargetMetadata(track, washInstrument);
      transitionActive = true;
      pendingRetune = true;
      transitionStartLevel = washEnvLevel;
      transitionOutSec = 4.5;
      transitionGapSec = 1.5;
      washPhase = 'transitionOut';
      washPhaseElapsed = 0;
      refreshActiveWashesRecord(track);
      return true;
    }

    transitionActive = false;
    applyWashEnvelopeGain(0.0, 0);
    retuneDroneBank(track, { forceAtZero: true });
    washesTriggered++;

    if (isContinuous) {
      currentWashCycleSec = fadeSec;
      washTimerSec = currentWashCycleSec;
      washPhase = 'fadeIn';
      washPhaseElapsed = 0;
      washStartFloor = 0.0;
      applyWashEnvelopeGain(0.001, 0.06);
    } else {
      const effectiveGap = gapSec > 0.05 ? gapSec : 3.0;
      currentWashCycleSec = fadeSec * 2 + sustainSec + effectiveGap;
      washTimerSec = currentWashCycleSec;
      washPhase = 'fadeIn';
      washPhaseElapsed = 0;
      washStartFloor = 0.0;
      applyWashEnvelopeGain(0.001, 0.06);
    }

    refreshActiveWashesRecord(track);
    return true;
  }

  // Smooth ~10s transition between tracks: fades out outgoing wash over 4.5s at locked pitch,
  // rests 1.5s in complete silence (retuning oscillators at 0.0 gain), and fades in new track over 4.2-4.5s.
  function crossfadeWashesToTrack(nextTrackObj, fadeSec = nextTrackObj?.washFade || 4.2) {
    applyTrackDroneDefaults(nextTrackObj);
    updateWashTargetMetadata(nextTrackObj, washInstrument);
    if (enabled) {
      triggerHeavenlyWash(nextTrackObj, { initial: false, crossfadeInSec: fadeSec });
    } else {
      applyWashEnvelopeGain(0.0, 0);
      retuneDroneBank(nextTrackObj, { forceAtZero: true });
    }
  }

  function tickWash(dt, track) {
    const fadeSec = Math.max(1.5, Math.min(10.0, Number(washFade) || 4.2));
    const sustainSec = Math.max(3.0, Math.min(12.0, Number(washSustain) || 7.0));
    const gapSec = Math.max(0, Math.min(8.0, Number(washGap) || 0));
    const isContinuous = (droneMode === 'Continuous drone') && (gapSec <= 0.05);

    // Handle long, smooth ~10s track transition phases ('transitionOut' -> 'transitionGap' -> 'fadeIn')
    if (washPhase === 'transitionOut') {
      washPhaseElapsed += dt;
      if (washPhaseElapsed >= transitionOutSec) {
        washPhase = 'transitionGap';
        washPhaseElapsed = 0;
        applyWashEnvelopeGain(0.0, 0);
        retuneDroneBank(track, { forceAtZero: true });
      } else {
        const u = Math.max(0, Math.min(1, washPhaseElapsed / transitionOutSec));
        const lvl = transitionStartLevel * 0.5 * (1.0 + Math.cos(Math.PI * u));
        applyWashEnvelopeGain(lvl, 0.06);
      }
      refreshActiveWashesRecord(track);
      return;
    }

    if (washPhase === 'transitionGap') {
      washPhaseElapsed += dt;
      applyWashEnvelopeGain(0.0, 0);
      if (pendingRetune) {
        retuneDroneBank(track, { forceAtZero: true });
      }
      if (washPhaseElapsed >= transitionGapSec) {
        transitionActive = false;
        retuneDroneBank(track, { forceAtZero: true });
        washesTriggered++;
        washPhase = 'fadeIn';
        washPhaseElapsed = 0;
        washStartFloor = 0.0;
        applyWashEnvelopeGain(0.001, 0.06);
      }
      refreshActiveWashesRecord(track);
      return;
    }

    if (isContinuous) {
      if (washPhase === 'gap' || washPhase === 'fadeOut' || washPhase === 'sustain') {
        if (washEnvLevel >= 0.98) {
          washPhase = 'continuous';
          washPhaseElapsed = 0;
          applyWashEnvelopeGain(1.0, 0.10);
        } else {
          washStartFloor = washEnvLevel;
          washPhase = 'fadeIn';
          washPhaseElapsed = 0;
        }
      }
      if (washPhase === 'fadeIn') {
        washPhaseElapsed += dt;
        if (washPhaseElapsed >= fadeSec) {
          washPhase = 'continuous';
          washPhaseElapsed = 0;
          applyWashEnvelopeGain(1.0, 0.10);
        } else {
          const u = Math.max(0, Math.min(1, washPhaseElapsed / fadeSec));
          const lvl = washStartFloor + (1.0 - washStartFloor) * 0.5 * (1.0 - Math.cos(Math.PI * u));
          applyWashEnvelopeGain(Math.max(0.001, lvl), 0.08);
        }
      } else {
        washPhase = 'continuous';
        washPhaseElapsed += dt;
        applyWashEnvelopeGain(1.0, 0.12);
      }
    } else {
      // Breathing wash mode: smooth cosine fade-in (4.0-4.5s) -> warm sustain (6.8-7.2s) -> cosine fade-out (4.0-4.5s) -> silent gap (2.5-3.2s)
      const effectiveGap = gapSec > 0.05 ? gapSec : 3.0;
      currentWashCycleSec = fadeSec * 2 + sustainSec + effectiveGap;

      if (washPhase === 'continuous') {
        washPhase = 'sustain';
        washPhaseElapsed = 0;
      }

      if (washPhase === 'fadeIn') {
        washPhaseElapsed += dt;
        if (washPhaseElapsed >= fadeSec) {
          washPhase = 'sustain';
          washPhaseElapsed = 0;
          applyWashEnvelopeGain(1.0, 0.08);
        } else {
          const u = Math.max(0, Math.min(1, washPhaseElapsed / fadeSec));
          const lvl = 0.5 * (1.0 - Math.cos(Math.PI * u));
          applyWashEnvelopeGain(Math.max(0.001, lvl), 0.08);
        }
      } else if (washPhase === 'sustain') {
        washPhaseElapsed += dt;
        applyWashEnvelopeGain(1.0, 0.10);
        if (washPhaseElapsed >= sustainSec) {
          washPhase = 'fadeOut';
          washPhaseElapsed = 0;
        }
      } else if (washPhase === 'fadeOut') {
        washPhaseElapsed += dt;
        if (washPhaseElapsed >= fadeSec) {
          washPhase = 'gap';
          washPhaseElapsed = 0;
          applyWashEnvelopeGain(0.0, 0);
          if (pendingRetune) {
            retuneDroneBank(track, { forceAtZero: true });
          }
        } else {
          const u = Math.max(0, Math.min(1, washPhaseElapsed / fadeSec));
          const lvl = 0.5 * (1.0 + Math.cos(Math.PI * u));
          applyWashEnvelopeGain(lvl, 0.08);
        }
      } else if (washPhase === 'gap') {
        washPhaseElapsed += dt;
        applyWashEnvelopeGain(0.0, 0);
        if (pendingRetune) {
          retuneDroneBank(track, { forceAtZero: true });
        }
        if (washPhaseElapsed >= effectiveGap) {
          retuneDroneBank(track, { forceAtZero: true });
          washesTriggered++;
          washPhase = 'fadeIn';
          washPhaseElapsed = 0;
          applyWashEnvelopeGain(0.001, 0.06);
        }
      }
    }

    refreshActiveWashesRecord(track);
  }

  function getWashRms() {
    if (!enabled || washEnvLevel <= 0.0005 || washLevel <= 0.0005) return 0;
    // Steady orthogonal harmonic RMS scaled by washEnvLevel and washLevel
    let sumSq = 0;
    for (let i = 0; i < droneBank.length; i++) {
      const g = droneBank[i].gain * 1.45;
      sumSq += 0.5 * g * g;
    }
    const baseRms = Math.sqrt(sumSq > 0 ? sumSq : 0.04);
    let liveRms = 0;
    if (washAnalyser && washAnalyserBuf) {
      try {
        washAnalyser.getFloatTimeDomainData(washAnalyserBuf);
        let acc = 0;
        for (let i = 0; i < washAnalyserBuf.length; i++) {
          acc += washAnalyserBuf[i] * washAnalyserBuf[i];
        }
        liveRms = Math.sqrt(acc / washAnalyserBuf.length);
      } catch {}
    }
    const synthRms = baseRms * washEnvLevel * washLevel;
    return Number((liveRms > 0.002 ? (0.35 * liveRms * washLevel + 0.65 * synthRms) : synthRms).toFixed(5));
  }

  function getPianoRms() {
    if (!enabled || !pianoAnalyser || !pianoAnalyserBuf) return 0;
    try {
      pianoAnalyser.getFloatTimeDomainData(pianoAnalyserBuf);
      let acc = 0;
      for (let i = 0; i < pianoAnalyserBuf.length; i++) {
        acc += pianoAnalyserBuf[i] * pianoAnalyserBuf[i];
      }
      return Number(Math.sqrt(acc / pianoAnalyserBuf.length).toFixed(5));
    } catch {
      return 0;
    }
  }

  function getRmsLevel() {
    return Number(Math.hypot(getPianoRms(), getWashRms()).toFixed(5));
  }

  function findClosestSample(voice, midiNote) {
    const list = sampleBank.get(voice) || (
      voice === 'piano'
        ? null
        : (sampleBank.get('pad') || sampleBank.get('choir') || sampleBank.get('drone') || sampleBank.get('tape'))
    );
    if (!list || !list.length) return null;
    let best = list[0];
    let bestDist = Math.abs(midiNote - best.midi);
    for (let i = 1; i < list.length; i++) {
      const d = Math.abs(midiNote - list[i].midi);
      if (d < bestDist) {
        bestDist = d;
        best = list[i];
      }
    }
    return best;
  }

  function scaleDegreeToMidi(track, degree, octaveOffset = 0) {
    const intervals = track.intervals;
    const len = intervals.length;
    const degIdx = ((degree % len) + len) % len;
    const octWrap = Math.floor(degree / len);
    return track.rootMidi + intervals[degIdx] + (octaveOffset + octWrap) * 12;
  }

  // Enso Markov consonance + chord-tone gravity pitch selector (from probability-engine.js & phrase-engine.js)
  function pickEnsoDegree(track) {
    const intervals = track.intervals;
    const baseWeights = track.degreeWeights;
    const gravity = track.markovGravity || [];
    const lastDeg = noteHistory.length > 0 ? noteHistory[0] : null;
    const lastInterval = lastDeg !== null && intervals[lastDeg] !== undefined ? intervals[lastDeg] : 0;

    const weights = intervals.map((iv, i) => {
      let w = Math.max(0.02, baseWeights[i] ?? 0.5);
      if (typeof gravity[i] === 'number') w *= gravity[i];
      if (currentChordDegrees && currentChordDegrees.includes(i)) w *= 1.35;
      if (lastDeg !== null) {
        const diff = Math.abs(iv - lastInterval) % 12;
        if (diff === 7) w *= 1.45;
        else if (diff === 5 || diff === 4 || diff === 3) w *= 1.32;
        else if (diff === 0 || diff === 2 || diff === 9) w *= 1.18;
        else if (diff === 1 || diff === 6) w *= 0.28;
      }
      return w;
    });

    let sum = 0;
    for (let i = 0; i < weights.length; i++) sum += weights[i];
    let r = Math.random() * sum;
    for (let i = 0; i < weights.length; i++) {
      r -= weights[i];
      if (r <= 0) {
        noteHistory.unshift(i);
        if (noteHistory.length > 8) noteHistory.length = 8;
        return i;
      }
    }
    return 0;
  }

  function triggerVoice(voice, midiNote, velocity = 0.5, pan = 0, gainScale = 1.0) {
    if (!enabled) return false;
    const audioCtx = ensureContext();
    if (!audioCtx) return false;
    if (audioCtx.state === 'suspended') {
      audioCtx.resume().catch(() => {});
    }

    const match = findClosestSample(voice, midiNote);
    if (!match || !match.buffer) return false;

    const now = audioCtx.currentTime;
    const src = audioCtx.createBufferSource();
    src.buffer = match.buffer;
    const rate = Math.pow(2, (midiNote - match.midi) / 12);
    src.playbackRate.setValueAtTime(rate, now);

    const track = ENSO_TRACKS[trackIndex];
    const isPiano = voice === 'piano';
    const vCfg = track.voices[voice] || { gain: isPiano ? 1.0 : 0.60, attack: track.globalAttackSec || 3.2, release: 8.0 };
    const instTrim = VOICE_INSTRUMENT_TRIM[voice] ?? 0.16;
    const env = audioCtx.createGain();

    const rawDur = match.buffer.duration / rate;
    const dur = Math.max(1.2, rawDur);
    const peak = Math.max(0.005, Math.min(0.85, velocity * (vCfg.gain || 0.65) * instTrim * gainScale));

    if (isPiano) {
      // Soft 8ms wool-felt hammer onset; hold full gain across the 11.5s–12.0s reverb-stamped sample
      // so the pre-baked Enso cathedral convolution tail rings out in full with 0% live reverb CPU!
      env.gain.setValueAtTime(0.0001, now);
      env.gain.linearRampToValueAtTime(peak, now + 0.008);
      env.gain.setValueAtTime(peak, now + Math.max(0.5, dur - 0.30));
      env.gain.linearRampToValueAtTime(0.0001, now + dur);
      pianoNotesTriggered++;
      lastPianoMidi = midiNote;
      lastPianoSampleMidi = match.midi;
      lastPianoPlaybackRate = Number(rate.toFixed(4));
      lastPianoPeakGain = Number(peak.toFixed(4));
      pianoNoteLog.push({ midi: midiNote, sampleMidi: match.midi, playbackRate: lastPianoPlaybackRate, peakGain: lastPianoPeakGain, source: match.source });
      if (pianoNoteLog.length > 64) pianoNoteLog.shift();
    } else if (voice === 'bell') {
      // Gentle glass chime bloom
      const att = Math.min(0.08, vCfg.attack || 0.06);
      env.gain.setValueAtTime(0.0001, now);
      env.gain.linearRampToValueAtTime(peak, now + att);
      env.gain.exponentialRampToValueAtTime(0.0001, now + dur);
    } else {
      // Slow, breathing Enso pad/choir/tape/drone swell (2.4s–4.5s fade-in, long release)
      const att = Math.min(dur * 0.42, Math.max(1.8, (vCfg.attack || track.globalAttackSec || 3.2) * 0.78));
      const sustainEnd = Math.min(dur * 0.62, att + 0.8);
      env.gain.setValueAtTime(0.0001, now);
      env.gain.linearRampToValueAtTime(peak, now + att);
      env.gain.setValueAtTime(peak * 0.94, now + sustainEnd);
      env.gain.exponentialRampToValueAtTime(0.0001, now + dur);
    }

    // Per-note warm lowpass damping when a piano sample is pitch-shifted up (rate > 1)
    let voiceNode = src;
    if (isPiano && rate > 1.02) {
      const noteLp = audioCtx.createBiquadFilter();
      noteLp.type = 'lowpass';
      noteLp.frequency.setValueAtTime(Math.max(760, (track.pianoCutoffHz || 1020) / Math.pow(rate, 0.45)), now);
      noteLp.Q.value = 0.5;
      src.connect(noteLp);
      voiceNode = noteLp;
    }

    const destBus = isPiano ? (pianoFeltFilter || dryBus) : (bgWarmFilter || dryBus);

    if (audioCtx.createStereoPanner) {
      const panner = audioCtx.createStereoPanner();
      panner.pan.setValueAtTime(Math.max(-1, Math.min(1, pan)), now);
      voiceNode.connect(env);
      env.connect(panner);
      panner.connect(destBus);
      if (!isPiano && fdnSend) panner.connect(fdnSend);
    } else {
      voiceNode.connect(env);
      env.connect(destBus);
      if (!isPiano && fdnSend) env.connect(fdnSend);
    }

    src.start(now);
    src.stop(now + dur + 0.05);
    notesTriggered++;
    return true;
  }

  function scheduleDelayed(fn, delayMs) {
    if (delayMs <= 4) {
      fn();
      return;
    }
    const id = setTimeout(() => {
      scheduledTimers.delete(id);
      if (enabled) fn();
    }, delayMs);
    scheduledTimers.add(id);
  }

  function clearScheduledTimers() {
    for (const id of scheduledTimers) clearTimeout(id);
    scheduledTimers.clear();
    bgVoiceTimers = [];
  }

  // Enso 3-Minute Phrase-Length LFO + Aleatoric Phrase Variation Engine (from phrase-engine.js)
  function getPhraseLengthLfoCutoff(track, totalSteps) {
    const lfo = track.phraseLengthLfo;
    if (!lfo || !lfo.enabled) return totalSteps;
    const period = lfo.periodSec || 180;
    const minR = lfo.minRatio ?? 0.50;
    const maxR = lfo.maxRatio ?? 1.00;
    const phase = ((totalListenedSec % period) + period) / period;
    const wave = 0.5 * (1 + Math.cos(2 * Math.PI * phase));
    const ratio = minR + (maxR - minR) * wave;
    const minSteps = Math.min(totalSteps, Math.max(2, Math.round(totalSteps * minR)));
    return Math.max(minSteps, Math.min(totalSteps, Math.round(totalSteps * ratio)));
  }

  function triggerTrackPhrase(track, { forceFull = false } = {}) {
    const phrases = track.phrases;
    if (!phrases || !phrases.length) return;
    phraseCycleCount++;

    // Alternate between primary and variant phrase patterns like eno_soundscape/js/phrase-engine.js
    const isPrimary = phraseCycleCount <= 1 || Math.random() < 0.55;
    const pattern = isPrimary ? phrases[0] : (phrases[1] || phrases[0]);
    const totalSteps = pattern.length;

    const lfoCutoff = forceFull ? totalSteps : getPhraseLengthLfoCutoff(track, totalSteps);
    let cutoffIdx = lfoCutoff;
    const omittedSteps = new Set();

    // Enso aleatoric phrase variations (start-only dice or 1–2 omitted inner notes for spacious breathing)
    if (!forceFull && phraseCycleCount > 1 && lfoCutoff >= 3 && Math.random() < 0.55) {
      if (Math.random() < 0.42) {
        cutoffIdx = 2 + Math.floor(Math.random() * Math.max(1, lfoCutoff - 2));
      } else {
        const omitCount = lfoCutoff >= 6 ? (1 + Math.floor(Math.random() * 2)) : 1;
        for (let k = 0; k < omitCount; k++) {
          const candidate = 1 + Math.floor(Math.random() * (lfoCutoff - 1));
          if (lfoCutoff - (omittedSteps.size + 1) >= 2) omittedSteps.add(candidate);
        }
      }
    }

    const speed = Math.max(0.5, track.rotationSpeed || 1.0);

    for (let sIdx = 0; sIdx < cutoffIdx; sIdx++) {
      if (omittedSteps.has(sIdx)) continue;
      const step = pattern[sIdx];
      const delayMs = Math.round((step.delayMs || 0) / speed);

      scheduleDelayed(() => {
        const degs = step.deg;
        const octs = step.oct || [];
        const rollMs = step.rollMs || track.chordRollMs || 110;
        // Polyphonic chord headroom scaling from eno_soundscape/js/phrase-engine.js
        const chordScale = degs.length > 1 ? (1 / Math.pow(degs.length, 0.24)) : 1.0;
        const vel = (step.vel || 0.42) * chordScale;

        for (let i = 0; i < degs.length; i++) {
          let oct = octs[i] || 0;
          if (!forceFull && Math.random() < 0.03) oct += Math.random() < 0.5 ? 1 : -1;
          const midi = scaleDegreeToMidi(track, degs[i], oct);
          const pan = (Math.random() * 2 - 1) * 0.34;
          scheduleDelayed(() => {
            triggerVoice('piano', midi, vel, pan, 1.0);
          }, i * rollMs);
        }
      }, delayMs);
    }
  }

  // Enso Single-Instrument Chord Progression (triggered by track.chordInstrument on its prime cycle)
  function triggerTrackChord(track) {
    const chords = track.chords;
    if (!chords || !chords.length) return;
    const chord = chords[chordStepIdx % chords.length];
    chordStepIdx++;
    currentChordDegrees = chord.deg.slice();

    const voice = track.chordInstrument || 'pad';
    const rollMs = chord.rollMs || track.chordRollMs || 120;
    const isPianoChord = voice === 'piano';

    if (!isPianoChord) {
      if (bgVoiceTimers.length >= 2) return;
      bgVoiceTimers.push(6.5);
    }

    // Optionally omit 1 inner chord tone (phraseNoteGate = 0.22 in phrase-engine.js) so chords stay airy
    let indices = chord.deg.map((_, idx) => idx);
    if (indices.length >= 4 && Math.random() < 0.28) {
      const drop = 1 + Math.floor(Math.random() * (indices.length - 2));
      indices = indices.filter((idx) => idx !== drop);
    }

    const chordScale = 1 / Math.pow(Math.max(1, indices.length), 0.26);
    const vel = (chord.vel || track.chordVel || 0.46) * chordScale;
    const gainScale = isPianoChord ? 0.85 : 0.68;

    indices.forEach((cIdx, orderIdx) => {
      const midi = scaleDegreeToMidi(track, chord.deg[cIdx], chord.oct[cIdx] || 0);
      const pan = ((orderIdx / Math.max(1, indices.length - 1)) * 2 - 1) * 0.36;
      scheduleDelayed(() => {
        triggerVoice(voice, midi, vel, pan, gainScale);
      }, orderIdx * rollMs);
    });
  }

  function tick(dt) {
    if (!enabled || dt <= 0) return;
    const clampedDt = Math.min(dt, 0.25);
    totalListenedSec += clampedDt;
    if (bgVoiceTimers.length > 0) {
      bgVoiceTimers = bgVoiceTimers.map((t) => t - clampedDt).filter((t) => t > 0);
    }

    // 1. 3-Minute (180s) Circle-of-Fifths Automatic Track Rotation
    if (autoRotate) {
      trackElapsedSec += clampedDt;
      if (trackElapsedSec >= TRACK_ROTATE_SECONDS) {
        clearScheduledTimers();
        trackElapsedSec = 0;
        trackIndex = (trackIndex + 1) % ENSO_TRACKS.length;
        phraseCycleCount = 0;
        chordStepIdx = 0;
        const nextT = ENSO_TRACKS[trackIndex];
        syncTrackPianoFilter(nextT);
        crossfadeWashesToTrack(nextT, nextT.washFade || 4.2);
        onTrackChange?.(nextT.label, nextT);
        phraseTimerSec = transitionActive ? (transitionOutSec + transitionGapSec + 1.0) : 2.4;
      }
    }

    const track = ENSO_TRACKS[trackIndex];

    // 2. Persistent Harmonic Synth Drone / Wash Engine (100% phase-continuous WebAudio oscillators,
    //    zero .ogg sample looping or decaying background sample triggers)
    tickWash(clampedDt, track);

    // 3. Enso Felt Piano Phrase Engine (spacious 14s–24s breathing room, never overlapping)
    phraseTimerSec -= clampedDt;
    if (phraseTimerSec <= 0) {
      const baseCycle = (track.phraseCycleSec || 16.0) / Math.max(0.6, track.rotationSpeed || 1.0);
      phraseTimerSec = baseCycle * (0.90 + Math.random() * 0.30);
      if (Math.random() <= (track.phraseGateProb ?? 0.76)) {
        triggerTrackPhrase(track);
      }
    }
  }

  function setEnabled(nextEnabled) {
    const bool = !!nextEnabled;
    if (bool === enabled) return enabled;
    enabled = bool;
    if (enabled) {
      const audioCtx = ensureContext();
      if (audioCtx) {
        if (audioCtx.state === 'suspended') audioCtx.resume().catch(() => {});
        masterGain.gain.cancelScheduledValues(audioCtx.currentTime);
        masterGain.gain.setTargetAtTime(volume, audioCtx.currentTime, 0.10);
        if (washMasterGain) {
          washMasterGain.gain.cancelScheduledValues(audioCtx.currentTime);
          washMasterGain.gain.setTargetAtTime(washLevel, audioCtx.currentTime, 0.12);
        }
      }
      // Reset timers & phases so the opening phrase has a full 16s+ breathing window after it plays
      const track = ENSO_TRACKS[trackIndex];
      syncTrackPianoFilter(track);
      phraseTimerSec = (track.phraseCycleSec || 16.0) * 1.05;
      voicePhases.pad = 0.45;
      voicePhases.bell = 0.10;
      voicePhases.choir = 0.30;
      voicePhases.tape = 0.15;
      voicePhases.drone = 0.55;

      // Immediately start the Persistent Harmonic Synth Drone / Wash Engine
      triggerHeavenlyWash(track, { initial: true, crossfadeInSec: washFade });

      const fireInitial = () => {
        if (!enabled) return;
        const curTrack = ENSO_TRACKS[trackIndex];
        scheduleDelayed(() => {
          if (enabled) triggerTrackPhrase(curTrack, { forceFull: true });
        }, 280);
      };
      if (sampleBank.size > 0) {
        fireInitial();
      } else {
        scheduleDelayed(fireInitial, 90);
      }
    } else {
      clearScheduledTimers();
      activeWashes = [];
      applyWashEnvelopeGain(0.0, 0.08);
      if (ctx && masterGain) {
        masterGain.gain.cancelScheduledValues(ctx.currentTime);
        masterGain.gain.setTargetAtTime(0.0001, ctx.currentTime, 0.15);
      }
    }
    onStateChange?.(enabled);
    return enabled;
  }

  function setVolume(nextVol) {
    volume = Math.max(0, Math.min(1, Number(nextVol) || 0));
    if (ctx && masterGain && enabled) {
      masterGain.gain.setTargetAtTime(volume, ctx.currentTime, 0.04);
    }
    return volume;
  }

  function setWashLevel(nextLevel) {
    const parsed = Number(nextLevel);
    washLevel = Number.isFinite(parsed) ? Math.max(0, Math.min(1, parsed)) : 0.33;
    if (ctx && washMasterGain) {
      washMasterGain.gain.setTargetAtTime(washLevel, ctx.currentTime, 0.06);
    }
    return washLevel;
  }

  function setDroneMode(nextMode) {
    if (ENSO_DRONE_MODES.includes(nextMode)) {
      droneMode = nextMode;
      const track = ENSO_TRACKS[trackIndex];
      if (droneMode === 'Continuous drone') {
        if (washPhase !== 'continuous' && washPhase !== 'fadeIn') {
          washStartFloor = Math.max(0.1, washEnvLevel);
          washPhase = washEnvLevel >= 0.95 ? 'continuous' : 'fadeIn';
          washPhaseElapsed = 0;
        }
      } else if (washPhase === 'continuous') {
        washPhase = 'sustain';
        washPhaseElapsed = 0;
      }
      refreshActiveWashesRecord(track);
    }
    return droneMode;
  }

  function setWashInstrument(nextInst) {
    if (ENSO_WASH_INSTRUMENTS.includes(nextInst)) {
      washInstrument = nextInst;
      const track = ENSO_TRACKS[trackIndex];
      const { filterCfg } = updateWashTargetMetadata(track, washInstrument);
      if (ctx && washSilkFilter && washFormantFilter && washWarmShelf) {
        const now = ctx.currentTime;
        washSilkFilter.frequency.setTargetAtTime(filterCfg.cutoffHz, now, 0.25);
        washSilkFilter.Q.setTargetAtTime(filterCfg.q, now, 0.25);
        washFormantFilter.frequency.setTargetAtTime(filterCfg.formantHz, now, 0.25);
        washFormantFilter.gain.setTargetAtTime(filterCfg.formantGainDb, now, 0.25);
        washWarmShelf.gain.setTargetAtTime(filterCfg.shelfDb, now, 0.25);
      }
      retuneDroneBank(track);
      refreshActiveWashesRecord(track);
    }
    return washInstrument;
  }

  function setWashFade(nextFade) {
    const parsed = Number(nextFade);
    if (Number.isFinite(parsed)) {
      washFade = Math.max(1.5, Math.min(10.0, parsed));
    }
    return washFade;
  }

  function setWashGap(nextGap) {
    const parsed = Number(nextGap);
    if (Number.isFinite(parsed)) {
      washGap = Math.max(0, Math.min(8.0, parsed));
    }
    return washGap;
  }

  function setTrack(labelOrId) {
    const q = String(labelOrId || '').trim().toLowerCase();
    const idx = ENSO_TRACKS.findIndex(
      (t) => t.label.toLowerCase() === q || t.id.toLowerCase() === q || t.name.toLowerCase() === q
    );
    if (idx >= 0) {
      const changed = idx !== trackIndex;
      trackIndex = idx;
      const track = ENSO_TRACKS[trackIndex];
      if (changed) {
        clearScheduledTimers();
        trackElapsedSec = 0;
        phraseCycleCount = 0;
        chordStepIdx = 0;
        syncTrackPianoFilter(track);
        crossfadeWashesToTrack(track, track.washFade || 4.2);
        onTrackChange?.(track.label, track);
        if (enabled) {
          if (transitionActive) {
            phraseTimerSec = transitionOutSec + transitionGapSec + 1.0;
          } else {
            phraseTimerSec = (track.phraseCycleSec || 16.0) * 1.05;
            scheduleDelayed(() => {
              if (enabled) triggerTrackPhrase(track, { forceFull: true });
            }, 260);
          }
        } else {
          phraseTimerSec = (track.phraseCycleSec || 16.0) * 1.05;
        }
      } else {
        applyTrackDroneDefaults(track);
        updateWashTargetMetadata(track, washInstrument);
        if (!enabled || washEnvLevel <= 0.0005) {
          retuneDroneBank(track, { forceAtZero: true });
        }
        onTrackChange?.(track.label, track);
      }
    }
    return ENSO_TRACKS[trackIndex];
  }

  retuneDroneBank(ENSO_TRACKS[trackIndex], { forceAtZero: true });

  function setAutoRotate(bool) {
    autoRotate = !!bool;
    return autoRotate;
  }

  function setReverbMode(mode) {
    if (ENSO_REVERB_MODES.includes(mode)) {
      reverbMode = mode;
      syncReverbMode();
    }
    return reverbMode;
  }

  function nextTrack(dir = 1) {
    clearScheduledTimers();
    trackIndex = (trackIndex + dir + ENSO_TRACKS.length) % ENSO_TRACKS.length;
    trackElapsedSec = 0;
    phraseCycleCount = 0;
    chordStepIdx = 0;
    const track = ENSO_TRACKS[trackIndex];
    syncTrackPianoFilter(track);
    crossfadeWashesToTrack(track, track.washFade || 4.2);
    onTrackChange?.(track.label, track);
    if (enabled) {
      if (transitionActive) {
        phraseTimerSec = transitionOutSec + transitionGapSec + 1.0;
      } else {
        phraseTimerSec = (track.phraseCycleSec || 16.0) * 1.05;
        scheduleDelayed(() => {
          if (enabled) triggerTrackPhrase(track, { forceFull: true });
        }, 260);
      }
    } else {
      phraseTimerSec = (track.phraseCycleSec || 16.0) * 1.05;
    }
    return track;
  }

  return {
    get enabled() { return enabled; },
    get volume() { return volume; },
    get washLevel() { return washLevel; },
    get droneMode() { return droneMode; },
    get washInstrument() { return washInstrument; },
    get washFade() { return washFade; },
    get washSustain() { return washSustain; },
    get washGap() { return washGap; },
    get washPhase() { return washPhase; },
    get washEnvLevel() { return Number(washEnvLevel.toFixed(4)); },
    get washRms() { return getWashRms(); },
    get rmsLevel() { return getRmsLevel(); },
    get pianoRms() { return getPianoRms(); },
    get transitioning() { return Boolean(transitionActive); },
    get transitionDurationSec() { return Number((transitionOutSec + transitionGapSec + (ENSO_TRACKS[trackIndex]?.washFade || 4.2)).toFixed(2)); },
    get oscFrequencies() { return droneBank.map((n) => Number(n.osc.frequency.value.toFixed(4))); },
    get oscMidis() { return droneBank.map((n) => n.midi); },
    get vibratoRateHz() { return vibratoLfo ? Number(vibratoLfo.frequency.value.toFixed(2)) : VIBRATO_RATE_HZ; },
    get vibratoDepthCents() { return vibratoGain ? Number(vibratoGain.gain.value.toFixed(2)) : VIBRATO_DEPTH_CENTS; },
    get autoRotate() { return autoRotate; },
    get reverbMode() { return reverbMode; },
    get rotateIntervalSec() { return TRACK_ROTATE_SECONDS; },
    get trackIndex() { return trackIndex; },
    get currentTrack() { return ENSO_TRACKS[trackIndex]; },
    get trackElapsedSec() { return trackElapsedSec; },
    get pianoLevel() { return pianoLevel; },
    get notesTriggered() { return notesTriggered; },
    get pianoNotesTriggered() { return pianoNotesTriggered; },
    get lastPianoMidi() { return lastPianoMidi; },
    get lastPianoSampleMidi() { return lastPianoSampleMidi; },
    get lastPianoPlaybackRate() { return lastPianoPlaybackRate; },
    get lastPianoPeakGain() { return lastPianoPeakGain; },
    get pianoNoteHistory() { return pianoNoteLog.slice(); },
    get washesTriggered() { return washesTriggered; },
    get activeWashCount() { return activeWashes.length; },
    get washCycleSec() { return currentWashCycleSec; },
    get lastWashInstrument() { return lastWashInstrument; },
    get lastWashVoices() { return lastWashVoices.slice(); },
    get lastWashChordMidis() { return lastWashChordMidis.slice(); },
    get lastWashTrackId() { return lastWashTrackId; },
    get oggSamplesLoaded() { return oggSamplesLoaded; },
    get offlineFallbackReady() { return offlineFallbackReady; },
    get preferredFormat() { return preferredFormat; },
    get liveConvolverCount() { return 0; }, // Strictly 0 live ConvolverNodes on the main AudioContext
    get fdnNodeCount() { return fdnNodes.length; },
    get pianoCutoffHz() { return pianoFeltFilter ? pianoFeltFilter.frequency.value : (ENSO_TRACKS[trackIndex]?.pianoCutoffHz || 1050); },
    get pianoFilterQ() { return pianoFeltFilter ? pianoFeltFilter.Q.value : 0.5; },
    get activeBgVoices() { return bgVoiceTimers.length; },
    tracks: ENSO_TRACKS,
    sampleZones: SAMPLE_ZONES,
    setEnabled,
    toggle() { return setEnabled(!enabled); },
    setVolume,
    setWashLevel,
    setDroneMode,
    setWashInstrument,
    setWashFade,
    setWashGap,
    getWashRms,
    setTrack,
    setAutoRotate,
    setReverbMode,
    nextTrack,
    triggerVoice,
    triggerHeavenlyWash,
    tick,
  };
}

