// Enso Ambient Soundscape Engine for Crystal Transformer
// Ports the 6 modal Circle-of-Fifths ambient tracks, scales, phrases, and 6 voices
// (Felt Piano, CS-80 Ribbon Pad, DX7 Glass Bell, Spectral Choir, Revox Tape Loop, Sub-Bass Pedal)
// from Enso Soundscapes (//depot/google3/experimental/users/tribble/eno_soundscape).
//
// Ultra-low phone CPU architecture (zero live ConvolverNode FFT cost):
// 1. Pre-baked 320 kbps stereo .ogg (with 320 kbps .mp3 fallback for iOS WebKit) instrument
//    samples in samples/*.ogg that have the full stereo cathedral convolution reverb + Eno
//    octave shimmer + Salamander Grand Piano v3 felt EQ already stamped onto each sample.
// 2. Instant local OfflineAudioContext + stereo ConvolverNode fallback buffer generator so
//    the very first note plays immediately while .ogg samples stream in the background.
// 3. Optional lightweight 4-tap non-convolution Feedback Delay Network (FDN) using native
//    C++ DelayNode + BiquadFilterNode lowpass damping ('Stamped OGG (0% CPU)',
//    'Hybrid (OGG + Lite FDN)', 'Lite FDN Live').
// 4. Automatic Circle-of-Fifths track rotation every 3 minutes (180 seconds).

export const TRACK_ROTATE_SECONDS = 180;

export const ENSO_REVERB_MODES = [
  'Stamped OGG (0% CPU)',
  'Hybrid (OGG + Lite FDN)',
  'Lite FDN Live',
];

export const SAMPLE_ZONES = [
  { id: 'piano_c2', voice: 'piano', midi: 36, dur: 6.2 },
  { id: 'piano_fs2', voice: 'piano', midi: 42, dur: 6.2 },
  { id: 'piano_c3', voice: 'piano', midi: 48, dur: 6.0 },
  { id: 'piano_fs3', voice: 'piano', midi: 54, dur: 6.0 },
  { id: 'piano_c4', voice: 'piano', midi: 60, dur: 5.8 },
  { id: 'piano_fs4', voice: 'piano', midi: 66, dur: 5.6 },
  { id: 'piano_c5', voice: 'piano', midi: 72, dur: 5.4 },
  { id: 'piano_fs5', voice: 'piano', midi: 78, dur: 5.2 },
  { id: 'piano_c6', voice: 'piano', midi: 84, dur: 5.0 },

  { id: 'pad_c3', voice: 'pad', midi: 48, dur: 7.0 },
  { id: 'pad_g3', voice: 'pad', midi: 55, dur: 7.0 },
  { id: 'pad_c4', voice: 'pad', midi: 60, dur: 7.0 },
  { id: 'pad_g4', voice: 'pad', midi: 67, dur: 7.0 },

  { id: 'bell_c4', voice: 'bell', midi: 60, dur: 5.8 },
  { id: 'bell_g4', voice: 'bell', midi: 67, dur: 5.8 },
  { id: 'bell_c5', voice: 'bell', midi: 72, dur: 5.5 },
  { id: 'bell_g5', voice: 'bell', midi: 79, dur: 5.5 },

  { id: 'choir_c4', voice: 'choir', midi: 60, dur: 7.2 },
  { id: 'choir_g4', voice: 'choir', midi: 67, dur: 7.2 },
  { id: 'choir_c5', voice: 'choir', midi: 72, dur: 7.2 },

  { id: 'tape_c3', voice: 'tape', midi: 48, dur: 6.2 },
  { id: 'tape_g3', voice: 'tape', midi: 55, dur: 6.2 },
  { id: 'tape_c4', voice: 'tape', midi: 60, dur: 6.2 },

  { id: 'drone_c2', voice: 'drone', midi: 36, dur: 7.6 },
  { id: 'drone_g2', voice: 'drone', midi: 43, dur: 7.6 },
  { id: 'drone_c3', voice: 'drone', midi: 48, dur: 7.6 },
];

export const ENSO_TRACKS = [
  {
    id: 'thursday',
    name: 'Sunlit Atrium',
    label: 'Sunlit Atrium (G Mixolydian)',
    keyLabel: 'G Mixolydian',
    tonic: 'G',
    rootMidi: 55,
    intervals: [0, 2, 4, 5, 7, 9, 10],
    degreeWeights: [0.90, 0.75, 0.60, 0.30, 0.80, 0.50, 0.40],
    phraseCycleSec: 8.4,
    chordCycleSec: 13.5,
    chordInstrument: 'choir',
    voices: {
      pad:   { cycle: 16.3, prob: 0.72, gain: 0.42, octave: 0, pan: 0.35 },
      piano: { cycle: 6.7,  prob: 0.80, gain: 0.88, octave: 0, pan: 0.42 },
      bell:  { cycle: 13.1, prob: 0.42, gain: 0.36, octave: 1, pan: 0.60 },
      choir: { cycle: 11.3, prob: 0.78, gain: 0.44, octave: 1, pan: 0.48 },
      tape:  { cycle: 17.9, prob: 0.50, gain: 0.34, octave: 0, pan: 0.38 },
      drone: { cycle: 29.3, prob: 0.76, gain: 0.46, octave: -1, pan: 0.15 },
    },
    phrases: [
      [
        { deg: [0, 4], oct: [0, 1], delayMs: 0, vel: 0.82 },
        { deg: [2], oct: [1], delayMs: 340, vel: 0.68 },
        { deg: [1], oct: [1], delayMs: 690, vel: 0.60 },
        { deg: [4], oct: [1], delayMs: 1080, vel: 0.74 },
        { deg: [6], oct: [0], delayMs: 1740, vel: 0.56 },
        { deg: [5], oct: [0], delayMs: 2110, vel: 0.64 },
        { deg: [2], oct: [0], delayMs: 2520, vel: 0.54 },
        { deg: [0], oct: [1], delayMs: 3320, vel: 0.76 },
      ],
      [
        { deg: [4], oct: [1], delayMs: 0, vel: 0.76 },
        { deg: [2], oct: [1], delayMs: 380, vel: 0.65 },
        { deg: [1], oct: [1], delayMs: 810, vel: 0.58 },
        { deg: [0], oct: [1], delayMs: 1560, vel: 0.72 },
      ],
    ],
    chords: [
      { deg: [0, 2, 4], oct: [0, 0, 0], rollMs: 96, vel: 0.68 },
      { deg: [6, 1, 3], oct: [-1, 0, 0], rollMs: 118, vel: 0.62 },
      { deg: [3, 5, 0], oct: [0, 0, 1], rollMs: 128, vel: 0.66 },
      { deg: [4, 6, 1], oct: [0, 0, 1], rollMs: 108, vel: 0.60 },
    ],
  },
  {
    id: 'reflection',
    name: 'Luminous Drift',
    label: 'Luminous Drift (C Lydian)',
    keyLabel: 'C Lydian',
    tonic: 'C',
    rootMidi: 48,
    intervals: [0, 2, 4, 6, 7, 9, 11],
    degreeWeights: [0.98, 0.86, 0.42, 0.78, 0.92, 0.48, 0.32],
    phraseCycleSec: 9.6,
    chordCycleSec: 14.8,
    chordInstrument: 'piano',
    voices: {
      pad:   { cycle: 17.3, prob: 0.74, gain: 0.45, octave: 0, pan: 0.38 },
      piano: { cycle: 7.9,  prob: 0.70, gain: 0.86, octave: 1, pan: 0.45 },
      bell:  { cycle: 11.3, prob: 0.48, gain: 0.38, octave: 1, pan: 0.64 },
      choir: { cycle: 21.9, prob: 0.56, gain: 0.40, octave: 1, pan: 0.50 },
      tape:  { cycle: 13.7, prob: 0.60, gain: 0.36, octave: 0, pan: 0.40 },
      drone: { cycle: 27.1, prob: 0.82, gain: 0.48, octave: -1, pan: 0.15 },
    },
    phrases: [
      [
        { deg: [0], oct: [1], delayMs: 0, vel: 0.68 },
        { deg: [3], oct: [1], delayMs: 740, vel: 0.58 },
        { deg: [4], oct: [1], delayMs: 1620, vel: 0.64 },
        { deg: [1], oct: [1], delayMs: 2740, vel: 0.54 },
      ],
      [
        { deg: [0], oct: [1], delayMs: 0, vel: 0.66 },
        { deg: [3], oct: [1], delayMs: 790, vel: 0.58 },
        { deg: [6], oct: [1], delayMs: 1710, vel: 0.54 },
        { deg: [4], oct: [1], delayMs: 2820, vel: 0.62 },
      ],
    ],
    chords: [
      { deg: [0, 2, 4, 6], oct: [0, 1, 1, 1], rollMs: 162, vel: 0.66 },
      { deg: [1, 3, 5, 0], oct: [1, 1, 1, 2], rollMs: 178, vel: 0.58 },
      { deg: [6, 1, 3, 5], oct: [0, 1, 1, 1], rollMs: 192, vel: 0.54 },
      { deg: [0, 3, 4, 6], oct: [0, 1, 1, 1], rollMs: 168, vel: 0.64 },
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
    phraseCycleSec: 10.4,
    chordCycleSec: 15.6,
    chordInstrument: 'piano',
    voices: {
      pad:   { cycle: 21.9, prob: 0.78, gain: 0.46, octave: 0, pan: 0.38 },
      piano: { cycle: 10.3, prob: 0.72, gain: 0.85, octave: 0, pan: 0.52 },
      bell:  { cycle: 7.9,  prob: 0.58, gain: 0.42, octave: 1, pan: 0.66 },
      choir: { cycle: 25.1, prob: 0.66, gain: 0.46, octave: 1, pan: 0.60 },
      tape:  { cycle: 17.3, prob: 0.62, gain: 0.38, octave: 0, pan: 0.48 },
      drone: { cycle: 31.1, prob: 0.88, gain: 0.50, octave: -1, pan: 0.18 },
    },
    phrases: [
      [
        { deg: [0], oct: [1], delayMs: 0, vel: 0.78 },
        { deg: [1], oct: [1], delayMs: 790, vel: 0.60 },
        { deg: [3], oct: [1], delayMs: 1730, vel: 0.68 },
        { deg: [4], oct: [1], delayMs: 2830, vel: 0.74 },
        { deg: [6], oct: [1], delayMs: 4010, vel: 0.62 },
        { deg: [4], oct: [1], delayMs: 4990, vel: 0.70 },
        { deg: [2], oct: [1], delayMs: 6070, vel: 0.58 },
        { deg: [0], oct: [1], delayMs: 7330, vel: 0.76 },
      ],
      [
        { deg: [4], oct: [1], delayMs: 0, vel: 0.74 },
        { deg: [3], oct: [1], delayMs: 930, vel: 0.66 },
        { deg: [1], oct: [1], delayMs: 2010, vel: 0.62 },
        { deg: [0], oct: [1], delayMs: 3290, vel: 0.72 },
      ],
    ],
    chords: [
      { deg: [0, 2, 4, 1], oct: [0, 0, 0, 1], rollMs: 196, vel: 0.72 },
      { deg: [1, 3, 5, 0], oct: [0, 0, 0, 1], rollMs: 216, vel: 0.64 },
      { deg: [5, 0, 2, 4], oct: [-1, 0, 0, 0], rollMs: 204, vel: 0.60 },
      { deg: [4, 6, 1, 3], oct: [0, 0, 1, 1], rollMs: 224, vel: 0.68 },
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
    phraseCycleSec: 9.2,
    chordCycleSec: 14.2,
    chordInstrument: 'pad',
    voices: {
      pad:   { cycle: 19.1, prob: 0.82, gain: 0.48, octave: 0, pan: 0.36 },
      piano: { cycle: 8.9,  prob: 0.74, gain: 0.86, octave: 1, pan: 0.44 },
      bell:  { cycle: 14.9, prob: 0.65, gain: 0.42, octave: 1, pan: 0.62 },
      choir: { cycle: 23.7, prob: 0.70, gain: 0.46, octave: 1, pan: 0.52 },
      tape:  { cycle: 16.3, prob: 0.58, gain: 0.36, octave: 0, pan: 0.40 },
      drone: { cycle: 29.9, prob: 0.85, gain: 0.50, octave: -1, pan: 0.15 },
    },
    phrases: [
      [
        { deg: [0, 4], oct: [0, 1], delayMs: 0, vel: 0.80 },
        { deg: [1], oct: [1], delayMs: 590, vel: 0.62 },
        { deg: [2], oct: [1], delayMs: 1240, vel: 0.68 },
        { deg: [4], oct: [1], delayMs: 1960, vel: 0.75 },
        { deg: [5], oct: [1], delayMs: 3040, vel: 0.64 },
        { deg: [4], oct: [1], delayMs: 3720, vel: 0.70 },
        { deg: [2], oct: [1], delayMs: 4460, vel: 0.58 },
        { deg: [0], oct: [1], delayMs: 5480, vel: 0.76 },
      ],
      [
        { deg: [4], oct: [1], delayMs: 0, vel: 0.76 },
        { deg: [1], oct: [1], delayMs: 680, vel: 0.62 },
        { deg: [2], oct: [1], delayMs: 1520, vel: 0.68 },
        { deg: [0], oct: [1], delayMs: 2660, vel: 0.78 },
      ],
    ],
    chords: [
      { deg: [0, 2, 4, 6], oct: [0, 1, 1, 1], rollMs: 148, vel: 0.68 },
      { deg: [5, 0, 2, 4], oct: [0, 1, 1, 1], rollMs: 162, vel: 0.62 },
      { deg: [3, 5, 0, 2], oct: [0, 0, 1, 1], rollMs: 154, vel: 0.66 },
      { deg: [4, 6, 1, 3], oct: [0, 0, 1, 1], rollMs: 144, vel: 0.60 },
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
    phraseCycleSec: 7.4,
    chordCycleSec: 12.6,
    chordInstrument: 'pad',
    voices: {
      pad:   { cycle: 13.9, prob: 0.58, gain: 0.38, octave: 0, pan: 0.34 },
      piano: { cycle: 5.3,  prob: 0.90, gain: 0.94, octave: 0, pan: 0.40 },
      bell:  { cycle: 11.3, prob: 0.38, gain: 0.32, octave: 1, pan: 0.58 },
      choir: { cycle: 23.1, prob: 0.44, gain: 0.36, octave: 1, pan: 0.46 },
      tape:  { cycle: 9.7,  prob: 0.72, gain: 0.42, octave: 0, pan: 0.38 },
      drone: { cycle: 26.3, prob: 0.68, gain: 0.44, octave: -1, pan: 0.14 },
    },
    phrases: [
      [
        { deg: [0], oct: [1], delayMs: 0, vel: 0.62 },
        { deg: [2], oct: [1], delayMs: 470, vel: 0.52 },
        { deg: [1], oct: [1], delayMs: 1390, vel: 0.48 },
        { deg: [4], oct: [0], delayMs: 1910, vel: 0.58 },
      ],
      [
        { deg: [0], oct: [1], delayMs: 0, vel: 0.62 },
        { deg: [1], oct: [1], delayMs: 490, vel: 0.50 },
        { deg: [2], oct: [1], delayMs: 1010, vel: 0.56 },
        { deg: [4], oct: [1], delayMs: 1820, vel: 0.60 },
        { deg: [3], oct: [1], delayMs: 2380, vel: 0.52 },
        { deg: [1], oct: [1], delayMs: 2960, vel: 0.48 },
        { deg: [0], oct: [1], delayMs: 3810, vel: 0.58 },
      ],
    ],
    chords: [
      { deg: [0, 2, 4, 6], oct: [0, 0, 0, 0], rollMs: 68, vel: 0.58 },
      { deg: [3, 5, 0, 2], oct: [0, 0, 1, 1], rollMs: 78, vel: 0.52 },
      { deg: [6, 1, 3, 5], oct: [-1, 0, 0, 0], rollMs: 65, vel: 0.48 },
      { deg: [4, 6, 1, 3], oct: [0, 0, 1, 1], rollMs: 84, vel: 0.56 },
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
    phraseCycleSec: 8.2,
    chordCycleSec: 12.2,
    chordInstrument: 'tape',
    voices: {
      pad:   { cycle: 15.7, prob: 0.68, gain: 0.42, octave: 0, pan: 0.36 },
      piano: { cycle: 6.1,  prob: 0.78, gain: 0.88, octave: 0, pan: 0.44 },
      bell:  { cycle: 12.7, prob: 0.40, gain: 0.34, octave: 1, pan: 0.60 },
      choir: { cycle: 19.9, prob: 0.54, gain: 0.38, octave: 1, pan: 0.48 },
      tape:  { cycle: 8.3,  prob: 0.82, gain: 0.48, octave: 0, pan: 0.42 },
      drone: { cycle: 24.1, prob: 0.78, gain: 0.46, octave: -1, pan: 0.16 },
    },
    phrases: [
      [
        { deg: [0], oct: [1], delayMs: 0, vel: 0.72 },
        { deg: [2], oct: [1], delayMs: 610, vel: 0.60 },
        { deg: [4], oct: [1], delayMs: 1220, vel: 0.66 },
        { deg: [1], oct: [1], delayMs: 2440, vel: 0.56 },
      ],
      [
        { deg: [0], oct: [1], delayMs: 0, vel: 0.72 },
        { deg: [1], oct: [1], delayMs: 610, vel: 0.58 },
        { deg: [2], oct: [1], delayMs: 1220, vel: 0.64 },
        { deg: [4], oct: [1], delayMs: 1830, vel: 0.70 },
        { deg: [3], oct: [1], delayMs: 3050, vel: 0.60 },
        { deg: [1], oct: [1], delayMs: 3660, vel: 0.56 },
        { deg: [0], oct: [1], delayMs: 4880, vel: 0.68 },
      ],
    ],
    chords: [
      { deg: [0, 2, 4, 6], oct: [0, 0, 0, 0], rollMs: 128, vel: 0.66 },
      { deg: [5, 0, 2, 4], oct: [-1, 0, 0, 0], rollMs: 142, vel: 0.60 },
      { deg: [2, 4, 6, 1], oct: [0, 0, 0, 1], rollMs: 132, vel: 0.58 },
      { deg: [6, 1, 3, 5], oct: [-1, 0, 0, 0], rollMs: 148, vel: 0.64 },
    ],
  },
];

export const ENSO_TRACK_NAMES = ENSO_TRACKS.map((t) => t.label);

export function createEnsoSound({ onTrackChange, onStateChange } = {}) {
  let ctx = null;
  let masterGain = null;
  let dryBus = null;
  let fdnSend = null;
  let fdnNodes = [];
  let enabled = false;
  let volume = 0.75;
  let autoRotate = true;
  let reverbMode = ENSO_REVERB_MODES[0];
  let trackIndex = 0;
  let trackElapsedSec = 0;
  let phraseTimerSec = 0.8;
  let chordTimerSec = 3.2;
  let phraseStepIdx = 0;
  let chordStepIdx = 0;
  let notesTriggered = 0;
  let oggSamplesLoaded = 0;
  let offlineFallbackReady = false;
  let sampleLoadStarted = false;
  let preferredFormat = 'ogg';

  // Map<voice, Array<{ midi, buffer, source: 'ogg'|'mp3'|'offline' }>>
  const sampleBank = new Map();
  const scheduledTimers = new Set();

  // Staggered initial phases across the 6 voices so notes unfold immediately
  const voicePhases = {
    pad: 0.86,
    piano: 0.92,
    bell: 0.64,
    choir: 0.72,
    tape: 0.55,
    drone: 0.88,
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
      // Never overwrite a high-bitrate server .ogg/.mp3 sample with an offline fallback
      if (list[existingIdx].source !== 'offline' && sourceTag === 'offline') return;
      list[existingIdx] = entry;
    } else {
      list.push(entry);
      list.sort((a, b) => a.midi - b.midi);
    }
  }

  // Lightweight 4-tap non-convolution Schroeder / Feedback Delay Network (FDN)
  // Uses 4 prime-length DelayNodes + 2 allpass diffusers + 1800 Hz lowpass damping filter
  function buildLiteFdnReverb(audioCtx, destNode) {
    const input = audioCtx.createGain();
    input.gain.value = 0.0; // controlled by reverbMode

    const ap1 = audioCtx.createBiquadFilter();
    ap1.type = 'allpass';
    ap1.frequency.value = 740;
    ap1.Q.value = 0.7;

    const ap2 = audioCtx.createBiquadFilter();
    ap2.type = 'allpass';
    ap2.frequency.value = 1650;
    ap2.Q.value = 0.7;

    input.connect(ap1);
    ap1.connect(ap2);

    const delayTimes = [0.043, 0.067, 0.097, 0.139];
    const merger = audioCtx.createChannelMerger(2);
    const outGain = audioCtx.createGain();
    outGain.gain.value = 0.42;

    const nodes = [input, ap1, ap2, merger, outGain];
    const delays = [];

    for (let i = 0; i < delayTimes.length; i++) {
      const d = audioCtx.createDelay(0.25);
      d.delayTime.value = delayTimes[i];
      const damp = audioCtx.createBiquadFilter();
      damp.type = 'lowpass';
      damp.frequency.value = 1800;
      const fb = audioCtx.createGain();
      fb.gain.value = 0.76;

      ap2.connect(d);
      d.connect(damp);
      damp.connect(fb);
      d.connect(merger, 0, i % 2);

      delays.push({ d, damp, fb });
      nodes.push(d, damp, fb);
    }

    // Cross-couple adjacent delay lines for dense modal tail without ConvolverNode FFT
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
      reverbMode === 'Lite FDN Live' ? 0.52 :
      reverbMode === 'Hybrid (OGG + Lite FDN)' ? 0.18 :
      0.0;
    fdnSend.gain.setTargetAtTime(sendVal, ctx.currentTime, 0.04);
  }

  // Instant zero-network fallback: renders stereo convolution-reverb-stamped root buffers
  // locally via OfflineAudioContext + ConvolverNode in ~35ms while 320kbps .ogg files stream in.
  async function buildOfflineStampedFallback() {
    const OfflineCtx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
    if (!OfflineCtx) return;
    const quickZones = [
      { voice: 'piano', midi: 48, dur: 4.2 },
      { voice: 'piano', midi: 60, dur: 4.0 },
      { voice: 'piano', midi: 72, dur: 3.8 },
      { voice: 'pad',   midi: 55, dur: 4.6 },
      { voice: 'bell',  midi: 67, dur: 4.0 },
      { voice: 'choir', midi: 60, dur: 4.6 },
      { voice: 'tape',  midi: 55, dur: 4.2 },
      { voice: 'drone', midi: 43, dur: 4.8 },
    ];

    const rate = 24000;
    await Promise.all(quickZones.map(async (z) => {
      try {
        const frames = Math.floor(rate * z.dur);
        const octx = new OfflineCtx(2, frames, rate);
        const freq = 440 * Math.pow(2, (z.midi - 69) / 12);

        // Stereo cathedral impulse response inside OfflineAudioContext
        const irDur = Math.min(z.dur - 0.3, 3.4);
        const irLen = Math.floor(rate * irDur);
        const irBuf = octx.createBuffer(2, irLen, rate);
        const irL = irBuf.getChannelData(0);
        const irR = irBuf.getChannelData(1);
        let s = (z.midi * 2654435761) >>> 0;
        const rand = () => { s = (s * 1664525 + 1013904223) >>> 0; return (s / 4294967296) * 2 - 1; };
        const decay = 3.8 / irDur;
        for (let i = 0; i < irLen; i++) {
          const t = i / rate;
          const env = Math.min(1, t / 0.015) * Math.exp(-decay * t);
          irL[i] = rand() * env * 0.45;
          irR[i] = rand() * env * 0.45;
        }

        const conv = octx.createConvolver();
        conv.buffer = irBuf;
        const damp = octx.createBiquadFilter();
        damp.type = 'lowpass';
        damp.frequency.value = 3800;
        const wet = octx.createGain();
        wet.gain.value = 0.55;
        damp.connect(conv);
        conv.connect(wet);
        wet.connect(octx.destination);

        const env = octx.createGain();
        const isPerc = z.voice === 'piano' || z.voice === 'bell';
        env.gain.setValueAtTime(0.0001, 0);
        env.gain.linearRampToValueAtTime(0.65, isPerc ? 0.012 : 0.75);
        env.gain.exponentialRampToValueAtTime(0.0001, z.dur - 0.15);
        env.connect(octx.destination);
        env.connect(damp);

        const o1 = octx.createOscillator();
        o1.type = z.voice === 'pad' ? 'sawtooth' : z.voice === 'tape' ? 'triangle' : 'sine';
        o1.frequency.value = freq;
        const o2 = octx.createOscillator();
        o2.type = 'triangle';
        o2.frequency.value = z.voice === 'bell' ? freq * 2.76 : freq * 2;
        const g2 = octx.createGain();
        g2.gain.value = z.voice === 'bell' ? 0.22 : 0.25;
        const lp = octx.createBiquadFilter();
        lp.type = 'lowpass';
        lp.frequency.value = Math.min(8000, freq * 3.2);
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

    // Load middle-register priority zones first for instant studio-grade playback
    const priority = SAMPLE_ZONES.filter((z) => z.midi >= 48 && z.midi <= 72);
    const outer = SAMPLE_ZONES.filter((z) => z.midi < 48 || z.midi > 72);
    await Promise.all(priority.map(fetchZone));
    await Promise.all(outer.map(fetchZone));
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

    fdnSend = buildLiteFdnReverb(ctx, masterGain);
    syncReverbMode();

    buildOfflineStampedFallback();
    loadServerSamples();
    return ctx;
  }

  function findClosestSample(voice, midiNote) {
    const list = sampleBank.get(voice) || sampleBank.get('piano');
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

  function pickWeightedDegree(track) {
    const weights = track.degreeWeights;
    let sum = 0;
    for (let i = 0; i < weights.length; i++) sum += weights[i];
    let r = Math.random() * sum;
    for (let i = 0; i < weights.length; i++) {
      r -= weights[i];
      if (r <= 0) return i;
    }
    return 0;
  }

  function triggerVoice(voice, midiNote, velocity = 0.7, pan = 0) {
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
    const vCfg = track.voices[voice] || { gain: 0.5 };
    const env = audioCtx.createGain();
    const peak = Math.max(0.02, Math.min(1.0, velocity * (vCfg.gain || 0.5) * 0.72));
    const dur = match.buffer.duration / rate;

    env.gain.setValueAtTime(peak, now);
    env.gain.setValueAtTime(peak, now + Math.max(0.1, dur * 0.65));
    env.gain.exponentialRampToValueAtTime(0.0001, now + dur);

    if (audioCtx.createStereoPanner) {
      const panner = audioCtx.createStereoPanner();
      panner.pan.setValueAtTime(Math.max(-1, Math.min(1, pan)), now);
      src.connect(env);
      env.connect(panner);
      panner.connect(dryBus);
      if (fdnSend) panner.connect(fdnSend);
    } else {
      src.connect(env);
      env.connect(dryBus);
      if (fdnSend) env.connect(fdnSend);
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
  }

  function triggerTrackPhrase(track) {
    const phrases = track.phrases;
    if (!phrases || !phrases.length) return;
    const pattern = phrases[phraseStepIdx % phrases.length];
    phraseStepIdx++;

    for (const step of pattern) {
      if (Math.random() < 0.14) continue; // Enso probabilistic note gate
      scheduleDelayed(() => {
        for (let i = 0; i < step.deg.length; i++) {
          const midi = scaleDegreeToMidi(track, step.deg[i], step.oct[i] || 0);
          const pan = (Math.random() * 2 - 1) * 0.38;
          scheduleDelayed(() => {
            triggerVoice('piano', midi, step.vel || 0.68, pan);
          }, i * 85);
        }
      }, step.delayMs || 0);
    }
  }

  function triggerTrackChord(track) {
    const chords = track.chords;
    if (!chords || !chords.length) return;
    const chord = chords[chordStepIdx % chords.length];
    chordStepIdx++;
    const voice = track.chordInstrument || 'pad';
    const rollMs = chord.rollMs || 120;

    for (let i = 0; i < chord.deg.length; i++) {
      const midi = scaleDegreeToMidi(track, chord.deg[i], chord.oct[i] || 0);
      const pan = ((i / Math.max(1, chord.deg.length - 1)) * 2 - 1) * 0.45;
      scheduleDelayed(() => {
        triggerVoice(voice, midi, chord.vel || 0.62, pan);
      }, i * rollMs);
    }
  }

  function tick(dt) {
    if (!enabled || dt <= 0) return;
    const clampedDt = Math.min(dt, 0.25);
    const track = ENSO_TRACKS[trackIndex];

    // 1. 3-Minute (180s) Circle-of-Fifths Automatic Track Rotation
    if (autoRotate) {
      trackElapsedSec += clampedDt;
      if (trackElapsedSec >= TRACK_ROTATE_SECONDS) {
        trackElapsedSec = 0;
        trackIndex = (trackIndex + 1) % ENSO_TRACKS.length;
        phraseStepIdx = 0;
        chordStepIdx = 0;
        const nextTrack = ENSO_TRACKS[trackIndex];
        onTrackChange?.(nextTrack.label, nextTrack);
        triggerTrackChord(nextTrack);
      }
    }

    // 2. Enso Piano Phrase Sequencer
    phraseTimerSec -= clampedDt;
    if (phraseTimerSec <= 0) {
      phraseTimerSec = (track.phraseCycleSec || 8.5) * (0.85 + Math.random() * 0.35);
      if (Math.random() < 0.82) {
        triggerTrackPhrase(track);
      }
    }

    // 3. Enso Rolled Chord Progression Sequencer
    chordTimerSec -= clampedDt;
    if (chordTimerSec <= 0) {
      chordTimerSec = (track.chordCycleSec || 13.5) * (0.88 + Math.random() * 0.28);
      if (Math.random() < 0.86) {
        triggerTrackChord(track);
      }
    }

    // 4. Asynchronous Prime-Length Tape Loops across all 6 Enso voices
    for (const [voiceId, cfg] of Object.entries(track.voices)) {
      voicePhases[voiceId] = (voicePhases[voiceId] || 0) + clampedDt / cfg.cycle;
      if (voicePhases[voiceId] >= 1.0) {
        voicePhases[voiceId] -= Math.floor(voicePhases[voiceId]);
        if (Math.random() <= cfg.prob) {
          const deg = voiceId === 'drone' ? (Math.random() < 0.72 ? 0 : 4) : pickWeightedDegree(track);
          let oct = cfg.octave || 0;
          if (voiceId !== 'drone' && Math.random() < 0.10) {
            oct += Math.random() < 0.65 ? 1 : -1;
          }
          const midi = scaleDegreeToMidi(track, deg, oct);
          const pan = (Math.random() * 2 - 1) * (cfg.pan || 0.4);
          const vel = 0.55 + Math.random() * 0.30;
          triggerVoice(voiceId, midi, vel, pan);
        }
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
        masterGain.gain.setTargetAtTime(volume, audioCtx.currentTime, 0.08);
      }
      // Trigger an immediate welcoming chord + piano motif once buffers are ready
      const fireInitial = () => {
        if (!enabled) return;
        const track = ENSO_TRACKS[trackIndex];
        triggerTrackChord(track);
        scheduleDelayed(() => triggerTrackPhrase(track), 420);
      };
      if (sampleBank.size > 0) {
        fireInitial();
      } else {
        scheduleDelayed(fireInitial, 90);
      }
    } else {
      clearScheduledTimers();
      if (ctx && masterGain) {
        masterGain.gain.cancelScheduledValues(ctx.currentTime);
        masterGain.gain.setTargetAtTime(0.0001, ctx.currentTime, 0.12);
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

  function setTrack(labelOrId) {
    const q = String(labelOrId || '').trim().toLowerCase();
    const idx = ENSO_TRACKS.findIndex(
      (t) => t.label.toLowerCase() === q || t.id.toLowerCase() === q || t.name.toLowerCase() === q
    );
    if (idx >= 0 && idx !== trackIndex) {
      trackIndex = idx;
      trackElapsedSec = 0;
      phraseStepIdx = 0;
      chordStepIdx = 0;
      if (enabled) {
        triggerTrackChord(ENSO_TRACKS[trackIndex]);
      }
    }
    return ENSO_TRACKS[trackIndex];
  }

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
    trackIndex = (trackIndex + dir + ENSO_TRACKS.length) % ENSO_TRACKS.length;
    trackElapsedSec = 0;
    phraseStepIdx = 0;
    chordStepIdx = 0;
    const track = ENSO_TRACKS[trackIndex];
    onTrackChange?.(track.label, track);
    if (enabled) triggerTrackChord(track);
    return track;
  }

  return {
    get enabled() { return enabled; },
    get volume() { return volume; },
    get autoRotate() { return autoRotate; },
    get reverbMode() { return reverbMode; },
    get rotateIntervalSec() { return TRACK_ROTATE_SECONDS; },
    get trackIndex() { return trackIndex; },
    get currentTrack() { return ENSO_TRACKS[trackIndex]; },
    get trackElapsedSec() { return trackElapsedSec; },
    get notesTriggered() { return notesTriggered; },
    get oggSamplesLoaded() { return oggSamplesLoaded; },
    get offlineFallbackReady() { return offlineFallbackReady; },
    get preferredFormat() { return preferredFormat; },
    get liveConvolverCount() { return 0; }, // Strictly 0 live ConvolverNodes on the main AudioContext
    get fdnNodeCount() { return fdnNodes.length; },
    tracks: ENSO_TRACKS,
    sampleZones: SAMPLE_ZONES,
    setEnabled,
    toggle() { return setEnabled(!enabled); },
    setVolume,
    setTrack,
    setAutoRotate,
    setReverbMode,
    nextTrack,
    triggerVoice,
    tick,
  };
}
