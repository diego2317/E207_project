package at.ofai.music.match;

import at.ofai.music.audio.FFT;
import at.ofai.music.match.GUI.FileNameSelection;
import at.ofai.music.util.Event;
import at.ofai.music.util.EventList;
import at.ofai.music.util.Format;
import at.ofai.music.util.Profile;
import at.ofai.music.util.WormEvent;
import at.ofai.music.worm.Plot;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.ListIterator;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.SourceDataLine;
import javax.sound.sampled.TargetDataLine;
import javax.sound.sampled.AudioFormat.Encoding;
import java.awt.GraphicsEnvironment;

public class PerformanceMatcher {
    protected PerformanceMatcher otherMatcher;
    protected boolean firstPM;
    protected AudioInputStream rawInputStream;
    protected AudioInputStream pcmInputStream;
    protected SourceDataLine audioOut;
    protected boolean audioOutputRequested;
    protected AudioFormat audioFormat;
    protected int channels;
    protected float sampleRate;
    protected String audioFileName;
    protected AudioFile audioFile;
    protected String matchFileName;
    protected String outputFileName;
    protected String featDumpFileName = null;
    protected at.ofai.music.match.PerformanceMatcher.PathType outputType;
    protected at.ofai.music.match.PerformanceMatcher.MetaType metadata;
    protected EventList events;
    protected WormHandler wormHandler;
    protected boolean liveWorm;
    protected double referenceFrequency;
    protected double matchFileOffset;
    protected boolean normalise1;
    protected boolean normalise2;
    protected boolean normalise3;
    protected boolean normalise4;
    protected boolean normalise5;
    protected boolean useSpectralDifference;
    protected boolean useChromaFrequencyMap;
    protected double scale;
    protected double hopTime;
    protected double fftTime;
    protected double blockTime;
    protected int hopSize;
    protected int fftSize;
    protected int blockSize;
    protected int frameCount;
    protected double frameRMS;
    protected double ltAverage;
    protected int runCount;
    protected boolean paused;
    protected int maxFrames;
    protected byte[] inputBuffer;
    protected double[] circBuffer;
    protected int cbIndex;
    protected double[] window;
    protected double[] reBuffer;
    protected double[] imBuffer;
    protected int[] freqMap;
    protected int freqMapSize;
    protected double[] prevFrame;
    protected double[] newFrame;
    protected double[][] frames;
    protected int[][] bestPathCost;
    protected byte[][] distance;
    protected int[] first;
    protected int[] last;
    protected boolean[] ignore;
    protected FileNameSelection progressCallback;
    protected long fileLength;
    protected static boolean silent = true;
    public static boolean batchMode = false;
    public static boolean matrixVisible = false;
    public static boolean guiVisible = true;
    public static boolean stop = false;
    public static final int liveInputBufferSize = 32768;
    public static final int outputBufferSize = 32768;
    protected static final int ADVANCE_THIS = 1;
    protected static final int ADVANCE_OTHER = 2;
    protected static final int ADVANCE_BOTH = 3;
    protected static final int MASK = 252;
    protected static final double decay = 0.99D;
    protected static final double silenceThreshold = 4.0E-4D;
    protected static final int MAX_RUN_COUNT = 3;
    protected static final int MAX_LENGTH = 3600;
    Plot plot = null;
    double[] plotX;
    double[] plotY;
    // $FF: synthetic field
    private static int[] $SWITCH_TABLE$at$ofai$music$match$PerformanceMatcher$PathType;

    public PerformanceMatcher(PerformanceMatcher var1) {
        this.otherMatcher = var1;
        this.firstPM = var1 == null;
        this.matchFileOffset = 0.0D;
        this.cbIndex = 0;
        this.frameRMS = 0.0D;
        this.ltAverage = 0.0D;
        this.frameCount = 0;
        this.runCount = 0;
        this.paused = false;
        this.hopSize = 0;
        this.fftSize = 0;
        this.blockSize = 0;
        this.hopTime = 512.0 / 22050.0;
        this.fftTime = 0.04644D;
        this.blockTime = 10.0D;
        this.normalise1 = true;
        this.normalise2 = false;
        this.normalise3 = false;
        this.normalise4 = true;
        this.normalise5 = false;
        this.useSpectralDifference = true;
        this.useChromaFrequencyMap = false;
        this.audioOutputRequested = false;
        this.scale = 90.0D;
        this.maxFrames = 0;
        this.progressCallback = null;
        this.metadata = at.ofai.music.match.PerformanceMatcher.MetaType.NONE;
        this.liveWorm = false;
        this.matchFileName = null;
        this.outputFileName = null;
        this.outputType = at.ofai.music.match.PerformanceMatcher.PathType.BACKWARD;
        this.events = null;
        this.referenceFrequency = 440.0D;
    }

    public void print() {
        System.err.println(this);
    }

    public String toString() {
        return "PerformanceMatcher\n\tAudio file: " + this.audioFileName + " ("
                + Format.d((double) (this.sampleRate / 1000.0F), 1).substring(1) + "kHz, " + this.channels
                + " channels)" + "\n\tHop size: " + this.hopSize + "\n\tFFT size: " + this.fftSize + "\n\tBlock size: "
                + this.blockSize;
    }

    public void setOtherMatcher(PerformanceMatcher var1) {
        this.otherMatcher = var1;
    }

    public void setProgressCallback(FileNameSelection var1) {
        this.progressCallback = var1;
    }

    public void setMatchFile(String var1, double var2, boolean var4) {
        this.matchFileName = var1;
        this.matchFileOffset = var2;
        this.metadata = var4 ? at.ofai.music.match.PerformanceMatcher.MetaType.WORM
                : at.ofai.music.match.PerformanceMatcher.MetaType.MATCH;

        try {
            if (var4) {
                this.setInputFile(EventList.getAudioFileFromWormFile(this.matchFileName));
                this.events = EventList.readWormFile(this.matchFileName);
                if (this.otherMatcher.wormHandler != null) {
                    this.otherMatcher.wormHandler.init();
                }
            } else if (var1.endsWith(".mid")) {
                this.events = EventList.readMidiFile(this.matchFileName);
                this.metadata = at.ofai.music.match.PerformanceMatcher.MetaType.MIDI;
            } else {
                this.events = EventList.readMatchFile(this.matchFileName);
            }
        } catch (Exception var6) {
            System.err.println("Error reading matchFile: " + var1 + "\n" + var6);
            this.events = null;
        }

    }

    public void setMatchFile(String var1, double var2) {
        this.setMatchFile(var1, var2, false);
    }

    public void setLabelFile(String var1) {
        this.matchFileName = var1;
        this.metadata = at.ofai.music.match.PerformanceMatcher.MetaType.LABEL;

        try {
            this.events = EventList.readLabelFile(var1);
        } catch (Exception var3) {
            System.err.println("Error reading labelFile: " + var1 + "\n" + var3);
            this.events = null;
        }

    }

    public void writeLabelFile(ScrollingMatrix var1) {
        EventList var2 = new EventList();
        AudioFile var3 = new AudioFile();
        if (MatrixFrame.useSmoothPath) {
            var3.setMatch(var1.sPathX, var1.hop2, var1.sPathY, var1.hop1, var1.sPathLength);
        } else {
            var3.setMatch(var1.bPathX, var1.hop2, var1.bPathY, var1.hop1, var1.bPathLength);
        }

        Iterator var5 = this.otherMatcher.events.l.iterator();

        while (var5.hasNext()) {
            Event var4 = (Event) var5.next();
            WormEvent var6 = new WormEvent(var3.fromReferenceTimeD(var4.keyDown), 0.0D, 0.0D, 0.0D, 0);
            var6.label = ((WormEvent) var4).label;
            var2.add(var6);
        }

        try {
            var2.writeLabelFile(this.matchFileName);
        } catch (Exception var7) {
            System.err.println("Unable to write output file: " + var7);
        }

    }

    public void dumpFeatures(String outPath) {
        if (outPath == null) return; // nothing to do

        try (java.io.BufferedWriter bw = new java.io.BufferedWriter(new java.io.FileWriter(outPath))) {

            // fresh state, but reuse the buffers already allocated in init()
            this.frameCount = 0;
            this.cbIndex = 0;
            java.util.Arrays.fill(this.prevFrame, 0.0);

            while (this.getFrame()) { // == read one hop
                /* === identical to the FIRST HALF of processFrame() === */
                for (int k = 0; k < fftSize; ++k) {
                    reBuffer[k] = window[k] * circBuffer[cbIndex];
                    if (++cbIndex == fftSize) cbIndex = 0;
                }
                java.util.Arrays.fill(imBuffer, 0.0);
                FFT.fft(reBuffer, imBuffer, -1);
                java.util.Arrays.fill(newFrame, 0.0);

                for (int k = 0; k <= fftSize/2; ++k) {
                    newFrame[freqMap[k]] += reBuffer[k] * reBuffer[k] 
                                        + imBuffer[k] * imBuffer[k];
                }

                /* optional spectral‑difference and normalisation -------- */
                double energy = 0.0;
                for (int b = 0; b < freqMapSize; ++b)
                    energy += newFrame[b];

                if (useSpectralDifference) {
                    for (int b = 0; b < freqMapSize; ++b) {
                        double v = newFrame[b];
                        newFrame[b] = (v > prevFrame[b]) ? v - prevFrame[b] : 0.0;
                    }
                }
                if (normalise1 && energy > 0.0) {
                    for (int b = 0; b < freqMapSize; ++b)
                        newFrame[b] /= energy;
                }

                /* -------------------------------------------------------- */

                // write one line: space‑separated doubles
                for (int b = 0; b < freqMapSize; ++b) {
                    bw.write(Double.toString(newFrame[b]));
                    bw.write(b == freqMapSize-1 ? '\n' : ' ');
                }

                // rotate buffers for next frame if spectral‑difference in use
                double[] tmp = prevFrame;
                prevFrame = newFrame;
                newFrame = tmp;
                ++frameCount;
            }

        } catch (Exception ex) {
            System.err.println("Feature‑dump failed: " + ex);
        }

        // rewind the matcher so alignment starts from scratch
        this.closeStreams();
        this.setInputFile(this.audioFileName);
    }

    public void writeMidiFile(ScrollingMatrix var1) {
        EventList var2 = new EventList();
        AudioFile var3 = new AudioFile();
        if (MatrixFrame.useSmoothPath) {
            var3.setMatch(var1.sPathX, var1.hop2, var1.sPathY, var1.hop1, var1.sPathLength);
        } else {
            var3.setMatch(var1.bPathX, var1.hop2, var1.bPathY, var1.hop1, var1.bPathLength);
        }

        Iterator var5 = this.otherMatcher.events.l.iterator();

        while (var5.hasNext()) {
            Event var4 = (Event) var5.next();
            Event var6 = var4.clone();
            var6.keyDown = var3.fromReferenceTimeD(var4.keyDown);
            var6.keyUp = var3.fromReferenceTimeD(var4.keyUp);
            var2.add(var6);
        }

        try {
            var2.writeMIDI(this.matchFileName);
        } catch (Exception var7) {
            System.err.println("Unable to write output file: " + var7);
        }

    }

    public void setLiveInput() {
        try {
            this.channels = 2;
            this.sampleRate = 22050.0F;
            AudioFormat var1 = new AudioFormat(Encoding.PCM_SIGNED, this.sampleRate, 16, this.channels,
                    this.channels * 2, this.sampleRate, false);
            TargetDataLine var2 = AudioSystem.getTargetDataLine(var1);
            var2.open(var1, 32768);
            this.pcmInputStream = new AudioInputStream(var2);
            this.audioFormat = this.pcmInputStream.getFormat();
            this.init();
            var2.start();
        } catch (Exception var3) {
            var3.printStackTrace();
            this.closeStreams();
        }

    }

    public void setInputFile(AudioFile var1) {
        this.audioFile = var1;
        this.setInputFile(var1.path);
    }

    public void setInputFile(String var1) {
        this.closeStreams();
        this.audioFileName = var1;

        try {
            if (this.audioFileName == null) {
                throw new Exception("No input file specified");
            }

            File var2 = new File(this.audioFileName);
            if (!var2.isFile()) {
                throw new FileNotFoundException("Requested file does not exist: " + this.audioFileName);
            }

            this.rawInputStream = AudioSystem.getAudioInputStream(var2);
            this.audioFormat = this.rawInputStream.getFormat();
            this.channels = this.audioFormat.getChannels();
            this.sampleRate = this.audioFormat.getSampleRate();
            this.pcmInputStream = this.rawInputStream;
            if (this.audioFormat.getEncoding() != Encoding.PCM_SIGNED
                    || this.audioFormat.getFrameSize() != this.channels * 2 || this.audioFormat.isBigEndian()) {
                AudioFormat var3 = new AudioFormat(Encoding.PCM_SIGNED, this.sampleRate, 16, this.channels,
                        this.channels * 2, this.sampleRate, false);
                this.pcmInputStream = AudioSystem.getAudioInputStream(var3, this.rawInputStream);
                this.audioFormat = var3;
            }

            this.init();
        } catch (Exception var4) {
            var4.printStackTrace();
            this.closeStreams();
        }

    }

    protected void init() {
        this.hopSize = (int) Math.round((double) this.sampleRate * this.hopTime);
        this.fftSize = (int) Math.round(Math.pow(2.0D,
                (double) Math.round(Math.log(this.fftTime * (double) this.sampleRate) / Math.log(2.0D))));
        this.blockSize = (int) Math.round(this.blockTime / this.hopTime);
        this.makeFreqMap(this.fftSize, this.sampleRate, this.referenceFrequency);
        int var1 = this.hopSize * this.channels * 2;
        if (this.inputBuffer == null || this.inputBuffer.length != var1) {
            this.inputBuffer = new byte[var1];
        }

        int var2;
        if (this.circBuffer == null || this.circBuffer.length != this.fftSize) {
            this.circBuffer = new double[this.fftSize];
            this.reBuffer = new double[this.fftSize];
            this.imBuffer = new double[this.fftSize];
            this.window = FFT.makeWindow(1, this.fftSize, this.fftSize);

            for (var2 = 0; var2 < this.fftSize; ++var2) {
                double[] var10000 = this.window;
                var10000[var2] *= Math.sqrt((double) this.fftSize);
            }
        }

        if (this.prevFrame != null && this.prevFrame.length == this.freqMapSize) {
            if (this.frames.length != this.blockSize) {
                this.frames = new double[this.blockSize][this.freqMapSize + 1];
            }
        } else {
            this.prevFrame = new double[this.freqMapSize];
            this.newFrame = new double[this.freqMapSize];
            this.frames = new double[this.blockSize][this.freqMapSize + 1];
        }

        var2 = (int) (3600.0D / this.hopTime);
        this.distance = new byte[var2][];
        this.bestPathCost = new int[var2][];
        this.first = new int[var2];
        this.last = new int[var2];
        if (this.normalise5 && this.events != null) {
            this.ignore = new boolean[var2];
            Arrays.fill(this.ignore, true);

            Event var3;
            for (Iterator var4 = this.events.l.iterator(); var4
                    .hasNext(); this.ignore[(int) Math.round(var3.keyDown / this.hopTime)] = false) {
                var3 = (Event) var4.next();
            }
        }

        for (int var6 = 0; var6 < this.blockSize; ++var6) {
            this.distance[var6] = new byte[4 * this.blockSize];
            this.bestPathCost[var6] = new int[4 * this.blockSize];
        }

        this.frameCount = 0;
        this.runCount = 0;
        this.cbIndex = 0;
        this.frameRMS = 0.0D;
        this.ltAverage = 0.0D;
        this.paused = false;
        this.progressCallback = null;
        if (this.pcmInputStream == this.rawInputStream) {
            this.fileLength = this.pcmInputStream.getFrameLength() / (long) this.hopSize;
        } else {
            this.fileLength = -1L;
        }

        if (!silent) {
            this.print();
        }

        try {
            if (this.audioOutputRequested) {
                this.audioOut = AudioSystem.getSourceDataLine(this.audioFormat);
                this.audioOut.open(this.audioFormat, 32768);
                this.audioOut.start();
            }
        } catch (Exception var5) {
            var5.printStackTrace();
            this.audioOut = null;
        }

    }

    public void closeStreams() {
        if (this.pcmInputStream != null) {
            try {
                this.pcmInputStream.close();
                if (this.pcmInputStream != this.rawInputStream) {
                    this.rawInputStream.close();
                }

                if (this.audioOut != null) {
                    this.audioOut.drain();
                    this.audioOut.close();
                }
            } catch (Exception var1) {
            }

            this.pcmInputStream = null;
            this.audioOut = null;
        }

    }

    protected void makeFreqMap(int var1, float var2, double var3) {
        this.freqMap = new int[var1 / 2 + 1];
        if (this.useChromaFrequencyMap) {
            this.makeChromaFrequencyMap(var1, var2, var3);
        } else {
            this.makeStandardFrequencyMap(var1, var2, var3);
        }

    }

    protected void makeStandardFrequencyMap(int var1, float var2, double var3) {
        double var5 = (double) (var2 / (float) var1);
        int var7 = (int) (2.0D / (Math.pow(2.0D, 0.08333333333333333D) - 1.0D));
        int var8 = (int) Math.round(Math.log((double) var7 * var5 / var3) / Math.log(2.0D) * 12.0D + 69.0D);

        int var9;
        for (var9 = 0; var9 <= var7; this.freqMap[var9++] = (int) Math.round((double) (var9 * 440) / var3)) {
        }

        double var10;
        for (; var9 <= var1 / 2; this.freqMap[var9++] = var7 + (int) Math.round(var10) - var8) {
            var10 = Math.log((double) var9 * var5 / var3) / Math.log(2.0D) * 12.0D + 69.0D;
            if (var10 > 127.0D) {
                var10 = 127.0D;
            }
        }

        this.freqMapSize = this.freqMap[var9 - 1] + 1;
        if (!silent) {
            System.err.println("Map size: " + this.freqMapSize + ";  Crossover at: " + var7);
            int var12 = Math.min(this.freqMap.length, 500);

            for (var9 = 0; var9 < var12; ++var9) {
                System.err.println("freqMap[" + var9 + "] = " + this.freqMap[var9]);
            }

            System.err.println("Reference frequency: " + var3);
        }

    }

    protected void makeChromaFrequencyMap(int var1, float var2, double var3) {
        double var5 = (double) (var2 / (float) var1);
        int var7 = (int) (1.0D / (Math.pow(2.0D, 0.08333333333333333D) - 1.0D));

        int var8;
        for (var8 = 0; var8 <= var7; this.freqMap[var8++] = 0) {
        }

        while (var8 <= var1 / 2) {
            double var9 = Math.log((double) var8 * var5 / var3) / Math.log(2.0D) * 12.0D + 69.0D;
            this.freqMap[var8++] = (int) Math.round(var9) % 12 + 1;
        }

        this.freqMapSize = 13;
    }

    public boolean getFrame() {
        if (this.pcmInputStream == null) {
            return false;
        } else {
            try {
                int var1 = this.pcmInputStream.read(this.inputBuffer);
                if (this.audioOut != null && var1 > 0 && this.audioOut.write(this.inputBuffer, 0, var1) != var1) {
                    System.err.println("Error writing to audio device");
                }

                if (var1 < this.inputBuffer.length) {
                    if (!silent) {
                        System.err.println("End of input: " + this.audioFileName);
                    }

                    this.closeStreams();
                    return false;
                }
            } catch (IOException var5) {
                var5.printStackTrace();
                this.closeStreams();
                return false;
            }

            this.frameRMS = 0.0D;
            int var3;
            double var6;
            label71: switch (this.channels) {
                case 1:
                    var3 = 0;

                    while (true) {
                        if (var3 >= this.inputBuffer.length) {
                            break label71;
                        }

                        var6 = (double) (this.inputBuffer[var3 + 1] << 8 | this.inputBuffer[var3] & 255) / 32768.0D;
                        this.frameRMS += var6 * var6;
                        this.circBuffer[this.cbIndex++] = var6;
                        if (this.cbIndex == this.fftSize) {
                            this.cbIndex = 0;
                        }

                        var3 += 2;
                    }
                case 2:
                    var3 = 0;

                    while (true) {
                        if (var3 >= this.inputBuffer.length) {
                            break label71;
                        }

                        var6 = (double) ((this.inputBuffer[var3 + 1] << 8 | this.inputBuffer[var3] & 255)
                                + (this.inputBuffer[var3 + 3] << 8 | this.inputBuffer[var3 + 2] & 255)) / 65536.0D;
                        this.frameRMS += var6 * var6;
                        this.circBuffer[this.cbIndex++] = var6;
                        if (this.cbIndex == this.fftSize) {
                            this.cbIndex = 0;
                        }

                        var3 += 4;
                    }
                default:
                    var3 = 0;

                    while (var3 < this.inputBuffer.length) {
                        var6 = 0.0D;

                        for (int var4 = 0; var4 < this.channels; var3 += 2) {
                            var6 += (double) (this.inputBuffer[var3 + 1] << 8 | this.inputBuffer[var3] & 255);
                            ++var4;
                        }

                        var6 /= 32768.0D * (double) this.channels;
                        this.frameRMS += var6 * var6;
                        this.circBuffer[this.cbIndex++] = var6;
                        if (this.cbIndex == this.fftSize) {
                            this.cbIndex = 0;
                        }
                    }
            }

            this.frameRMS = Math.sqrt(this.frameRMS / (double) this.inputBuffer.length);
            return true;
        }
    }

    protected void processFrame(boolean dump) {
        if (this.getFrame()) {
            int var1;
            for (var1 = 0; var1 < this.fftSize; ++var1) {
                this.reBuffer[var1] = this.window[var1] * this.circBuffer[this.cbIndex];
                if (++this.cbIndex == this.fftSize) {
                    this.cbIndex = 0;
                }
            }

            Arrays.fill(this.imBuffer, 0.0D);
            FFT.fft(this.reBuffer, this.imBuffer, -1);
            Arrays.fill(this.newFrame, 0.0D);

            double[] var10000;
            for (var1 = 0; var1 <= this.fftSize / 2; ++var1) {
                var10000 = this.newFrame;
                int var10001 = this.freqMap[var1];
                var10000[var10001] += this.reBuffer[var1] * this.reBuffer[var1]
                        + this.imBuffer[var1] * this.imBuffer[var1];
            }

            var1 = this.frameCount % this.blockSize;
            int var7;
            if (this.firstPM && this.frameCount >= this.blockSize) {
                int var2 = this.last[this.frameCount - this.blockSize] - this.first[this.frameCount - this.blockSize];
                byte[] var3 = this.distance[this.frameCount - this.blockSize];
                byte[] var4 = new byte[var2];
                int[] var5 = this.bestPathCost[this.frameCount - this.blockSize];
                int[] var6 = new int[var2];

                for (var7 = 0; var7 < var2; ++var7) {
                    var4[var7] = var3[var7];
                    var6[var7] = var5[var7];
                }

                this.distance[this.frameCount] = var3;
                this.distance[this.frameCount - this.blockSize] = var4;
                this.bestPathCost[this.frameCount] = var5;
                this.bestPathCost[this.frameCount - this.blockSize] = var6;
            }

            double var15 = 0.0D;
            int var16;
            if (this.useSpectralDifference) {
                for (var16 = 0; var16 < this.freqMapSize; ++var16) {
                    var15 += this.newFrame[var16];
                    if (this.newFrame[var16] > this.prevFrame[var16]) {
                        this.frames[var1][var16] = this.newFrame[var16] - this.prevFrame[var16];
                    } else {
                        this.frames[var1][var16] = 0.0D;
                    }
                }
            } else {
                for (var16 = 0; var16 < this.freqMapSize; ++var16) {
                    this.frames[var1][var16] = this.newFrame[var16];
                    var15 += this.frames[var1][var16];
                }
            }

            if (this.plot != null) {
                if (this.plotX == null) {
                    this.plotX = new double[this.freqMapSize];
                    this.plotY = new double[this.freqMapSize];

                    for (var16 = 0; var16 < this.freqMapSize; ++var16) {
                        this.plotX[var16] = (double) var16;
                    }

                    this.plot.addPlot(this.plotX, this.plotY);
                }

                for (var16 = 0; var16 < this.freqMapSize; ++var16) {
                    this.plotY[var16] = this.frames[var1][var16];
                }

                this.plot.update();
            }

            this.frames[var1][this.freqMapSize] = var15;
            if (this.wormHandler != null) {
                this.wormHandler.addPoint(var15);
            }

            double var18 = this.frameCount >= 200 ? 0.99D
                    : (this.frameCount < 100 ? 0.0D : (double) (this.frameCount - 100) / 100.0D);
            if (this.ltAverage == 0.0D) {
                this.ltAverage = var15;
            } else {
                this.ltAverage = this.ltAverage * var18 + var15 * (1.0D - var18);
            }

            int var17;
            if (var15 <= 4.0E-4D) {
                for (var17 = 0; var17 < this.freqMapSize; ++var17) {
                    this.frames[var1][var17] = 0.0D;
                }
            } else if (this.normalise1) {
                for (var17 = 0; var17 < this.freqMapSize; ++var17) {
                    var10000 = this.frames[var1];
                    var10000[var17] /= var15;
                }
            } else if (this.normalise3) {
                for (var17 = 0; var17 < this.freqMapSize; ++var17) {
                    var10000 = this.frames[var1];
                    var10000[var17] /= this.ltAverage;
                }
            }

            var17 = this.otherMatcher.frameCount;
            var7 = var17 - this.blockSize;
            if (var7 < 0) {
                var7 = 0;
            }

            this.first[this.frameCount] = var7;
            this.last[this.frameCount] = var17;
            boolean var8 = false;
            int var9 = -1;

            int var10;
            for (var10 = -1; var7 < var17; ++var7) {
                int var11 = this.calcDistance(this.frames[var1], this.otherMatcher.frames[var7 % this.blockSize]);
                if (this.ignore != null && this.ignore[this.frameCount]
                        || this.otherMatcher.ignore != null && this.otherMatcher.ignore[var7]) {
                    var11 /= 4;
                }

                if (var10 < 0) {
                    var9 = var11;
                    var10 = var11;
                } else if (var11 > var10) {
                    var10 = var11;
                } else if (var11 < var9) {
                    var9 = var11;
                }

                if (var11 >= 255) {
                    var8 = true;
                    var11 = 255;
                }

                if (this.frameCount == 0 && var7 == 0) {
                    this.setValue(0, 0, 0, 0, var11);
                } else if (this.frameCount == 0) {
                    this.setValue(0, var7, 2, this.getValue(0, var7 - 1, true), var11);
                } else if (var7 == 0) {
                    this.setValue(this.frameCount, var7, 1, this.getValue(this.frameCount - 1, 0, true), var11);
                } else {
                    int var12;
                    int var13;
                    if (var7 == this.otherMatcher.frameCount - this.blockSize) {
                        var12 = this.getValue(this.frameCount - 1, var7, true);
                        if (this.first[this.frameCount - 1] == var7) {
                            this.setValue(this.frameCount, var7, 1, var12, var11);
                        } else {
                            var13 = this.getValue(this.frameCount - 1, var7 - 1, true);
                            if (var13 + var11 <= var12) {
                                this.setValue(this.frameCount, var7, 3, var13, var11);
                            } else {
                                this.setValue(this.frameCount, var7, 1, var12, var11);
                            }
                        }
                    } else {
                        var12 = this.getValue(this.frameCount, var7 - 1, true);
                        var13 = this.getValue(this.frameCount - 1, var7, true);
                        int var14 = this.getValue(this.frameCount - 1, var7 - 1, true);
                        if (var12 <= var13) {
                            if (var14 + var11 <= var12) {
                                this.setValue(this.frameCount, var7, 3, var14, var11);
                            } else {
                                this.setValue(this.frameCount, var7, 2, var12, var11);
                            }
                        } else if (var14 + var11 <= var13) {
                            this.setValue(this.frameCount, var7, 3, var14, var11);
                        } else {
                            this.setValue(this.frameCount, var7, 1, var13, var11);
                        }
                    }
                }

                int var10002 = this.otherMatcher.last[var7]++;
            }

            double[] var19 = this.prevFrame;
            this.prevFrame = this.newFrame;
            this.newFrame = var19;
            ++this.frameCount;
            ++this.runCount;
            this.otherMatcher.runCount = 0;
            if (var8 && !silent) {
                System.err
                        .println("WARNING: overflow in distance metric: frame " + this.frameCount + ", val = " + var10);
            }

            if (this.frameCount % 100 == 0) {
                if (!silent) {
                    System.err.println("Progress:" + this.frameCount + " " + Format.d(this.ltAverage, 3));
                    Profile.report();
                }

                if (this.progressCallback != null && this.fileLength > 0L) {
                    this.progressCallback.setFraction((double) this.frameCount / (double) this.fileLength);
                }
            }

            if (this.frameCount == this.maxFrames) {
                this.closeStreams();
            }
            
            if (dump){
                for (var1 = 0; var1 <= this.newFrame.length - 1; ++var1) { // Only print chromas 1-12
                System.out.print(this.newFrame[var1] + ", ");
            }
            System.out.println("\n");
            }
            
            // Dump alignment coordinates - find best match for current frame
            // Note: Alignment is computed for both matchers automatically, but all alignment data
            // is stored in the first matcher (firstPM). Both var0 and var1 call processFrame,
            // and both compute alignment (lines 741-814), but data accumulates in var0's arrays.
            // Alignment starts at frame 0, so we dump from frame 0 onwards.
                // // Print matcher info on first frame
                // if (this.frameCount == 1) {
                //     System.err.println(this.toString());
                //     if (this.otherMatcher != null) {
                //         System.err.println(this.otherMatcher.toString());
                //     }
                // }
                
                // int frame1 = this.frameCount - 1;  // Current frame in first matcher (var0)
                // int bestMatchFrame = -1;
                // int minCost = Integer.MAX_VALUE;
                
                // // Check if alignment data exists for this frame
                // if (this.first[frame1] >= 0 && this.last[frame1] > this.first[frame1] && 
                //     this.bestPathCost[frame1] != null) {
                //     int firstIdx = this.first[frame1];
                //     int lastIdx = this.last[frame1];
                    
                //     // Use upper right corner of the alignment search rectangle
                //     // (frame1, lastIdx - 1) instead of minimum cost frame
                //     bestMatchFrame = lastIdx - 1;
                // }
                
                // System.out.print("\n" + "ALIGNMENT: " + this.frameCount + ", " + this.otherMatcher.frameCount + "\n");
            }
        }
        // System.err.println(this.toString());  // Shows full matcher info


    protected int calcDistance(double[] var1, double[] var2) {
        double var3 = 0.0D;
        double var5 = 0.0D;

        for (int var7 = 0; var7 < this.freqMapSize; ++var7) {
            var3 += Math.abs(var1[var7] - var2[var7]);
            var5 += var1[var7] + var2[var7];
        }

        if (var5 == 0.0D) {
            return 0;
        } else if (this.normalise2) {
            return (int) (this.scale * var3 / var5);
        } else if (!this.normalise4) {
            return (int) (this.scale * var3);
        } else {
            double var9 = (8.0D + Math.log(var5)) / 10.0D;
            if (var9 < 0.0D) {
                var9 = 0.0D;
            } else if (var9 > 1.0D) {
                var9 = 1.0D;
            }

            return (int) (this.scale * var3 / var5 * var9);
        }
    }

    protected int getValue(int var1, int var2, boolean var3) {
        return this.firstPM ? this.bestPathCost[var1][var2 - this.first[var1]]
                : this.otherMatcher.bestPathCost[var2][var1 - this.otherMatcher.first[var2]];
    }

    protected void setValue(int var1, int var2, int var3, int var4, int var5) {
        if (this.firstPM) {
            this.distance[var1][var2 - this.first[var1]] = (byte) (var5 & 252 | var3);
            this.bestPathCost[var1][var2 - this.first[var1]] = var4 + (var3 == 3 ? var5 * 2 : var5);
        } else {
            if (var3 == 1) {
                var3 = 2;
            } else if (var3 == 2) {
                var3 = 1;
            }

            int var6 = var1 - this.otherMatcher.first[var2];
            if (var6 == this.otherMatcher.distance[var2].length) {
                int[] var7 = new int[var6 * 2];
                byte[] var8 = new byte[var6 * 2];

                for (int var9 = 0; var9 < var6; ++var9) {
                    var7[var9] = this.otherMatcher.bestPathCost[var2][var9];
                    var8[var9] = this.otherMatcher.distance[var2][var9];
                }

                this.otherMatcher.bestPathCost[var2] = var7;
                this.otherMatcher.distance[var2] = var8;
            }

            this.otherMatcher.distance[var2][var6] = (byte) (var5 & 252 | var3);
            this.otherMatcher.bestPathCost[var2][var6] = var4 + (var3 == 3 ? var5 * 2 : var5);
        }

    }

    public LinkedList<at.ofai.music.match.PerformanceMatcher.Onset> evaluateMatch(PerformanceMatcher var1) {
        if (this.events != null && var1.events != null) {
            double var3 = Double.NaN;
            int var5 = 0;
            double var6 = 0.0D;
            double var8 = 0.0D;
            LinkedList var10 = new LinkedList();
            Iterator var11 = this.events.iterator();

            while (true) {
                Event var2;
                while (var11.hasNext()) {
                    var2 = (Event) var11.next();
                    if (var5 == 0) {
                        var6 = var2.keyDown;
                        if (this.metadata == at.ofai.music.match.PerformanceMatcher.MetaType.MATCH
                                || this.metadata == at.ofai.music.match.PerformanceMatcher.MetaType.MIDI) {
                            var8 = this.matchFileOffset - var2.keyDown;
                        }

                        var5 = 1;
                        var3 = var2.scoreBeat;
                    } else if (var2.scoreBeat == var3) {
                        var6 += var2.keyDown;
                        ++var5;
                    } else {
                        var10.add(new at.ofai.music.match.PerformanceMatcher.Onset(this, var3,
                                var6 / (double) var5 + var8, -1.0D));
                        var6 = var2.keyDown;
                        var5 = 1;
                        var3 = var2.scoreBeat;
                    }
                }

                if (var5 != 0) {
                    var10.add(new at.ofai.music.match.PerformanceMatcher.Onset(this, var3, var6 / (double) var5 + var8,
                            -1.0D));
                }

                if (var10.size() == 0) {
                    return null;
                }

                var5 = 0;
                var6 = 0.0D;
                var8 = 0.0D;
                ListIterator var14 = var10.listIterator();
                at.ofai.music.match.PerformanceMatcher.Onset var12 = (at.ofai.music.match.PerformanceMatcher.Onset) var14
                        .next();
                var3 = Double.NaN;
                Iterator var13 = var1.events.iterator();

                while (true) {
                    while (var13.hasNext()) {
                        var2 = (Event) var13.next();
                        if (var5 == 0) {
                            var6 = var2.keyDown;
                            if (var1.metadata == at.ofai.music.match.PerformanceMatcher.MetaType.MATCH
                                    || var1.metadata == at.ofai.music.match.PerformanceMatcher.MetaType.MIDI) {
                                var8 = var1.matchFileOffset - var2.keyDown;
                            }

                            var5 = 1;
                            var3 = var2.scoreBeat;
                        } else if (var2.scoreBeat == var3) {
                            var6 += var2.keyDown;
                            ++var5;
                        } else {
                            while (var12.beat < var3 && var14.hasNext()) {
                                var12 = (at.ofai.music.match.PerformanceMatcher.Onset) var14.next();
                            }

                            while (var12.beat > var3 && var14.hasPrevious()) {
                                var12 = (at.ofai.music.match.PerformanceMatcher.Onset) var14.previous();
                            }

                            if (var12.beat == var3) {
                                var12.time2 = var6 / (double) var5 + var8;
                            } else {
                                var14.add(new at.ofai.music.match.PerformanceMatcher.Onset(this, var3, -1.0D,
                                        var6 / (double) var5 + var8));
                            }

                            var6 = var2.keyDown;
                            var5 = 1;
                            var3 = var2.scoreBeat;
                        }
                    }

                    if (var5 != 0) {
                        while (var12.beat < var3 && var14.hasNext()) {
                            var12 = (at.ofai.music.match.PerformanceMatcher.Onset) var14.next();
                        }

                        while (var12.beat > var3 && var14.hasPrevious()) {
                            var12 = (at.ofai.music.match.PerformanceMatcher.Onset) var14.previous();
                        }

                        if (var12.beat == var3) {
                            var12.time2 = var6 / (double) var5 + var8;
                        } else {
                            var14.add(new at.ofai.music.match.PerformanceMatcher.Onset(this, var3, -1.0D,
                                    var6 / (double) var5 + var8));
                        }
                    }

                    var14 = var10.listIterator();

                    while (true) {
                        String var15;
                        do {
                            if (!var14.hasNext()) {
                                return var10;
                            }

                            var12 = (at.ofai.music.match.PerformanceMatcher.Onset) var14.next();
                            var15 = String.format("%8.3f %8.3f %8.3f", var12.beat, var12.time1, var12.time2);
                        } while (!(var12.time1 < 0.0D) && !(var12.time2 < 0.0D) && !(var12.beat < 0.0D));

                        System.err.println("Match Error: " + var15);
                        var14.remove();
                    }
                }
            }
        } else {
            return null;
        }
    }

    public static void doMatch(PerformanceMatcher var0, PerformanceMatcher var1, ScrollingMatrix var2) {
        Finder var3 = new Finder(var0, var1);

        // Only dump for var0 (first file)
        boolean dumpVar0 = true;
        boolean dumpVar1 = false;  // Never dump second file


        while (var0.pcmInputStream != null || var1.pcmInputStream != null) {
            if (var0.frameCount < var0.blockSize) {
                var0.processFrame(dumpVar0);
                var1.processFrame(dumpVar1);
            } else if (var0.pcmInputStream == null) {
                var1.processFrame(dumpVar1);
            } else if (var1.pcmInputStream == null) {
                var0.processFrame(dumpVar0);
            } else if (var0.paused) {
                if (var1.paused) {
                    try {
                        if (stop) {
                            break;
                        }

                        Thread.sleep(100L);
                        continue;
                    } catch (InterruptedException var4) {
                    }
                } else {
                    var1.processFrame(dumpVar1);
                }
            } else if (var1.paused) {
                var0.processFrame(dumpVar0);
            } else if (var0.runCount >= 3) {
                var1.processFrame(dumpVar1);
            } else if (var1.runCount >= 3) {
                var0.processFrame(dumpVar0);
            } else {
                switch (var3.getExpandDirection(var0.frameCount - 1, var1.frameCount - 1)) {
                    case 1:
                        var0.processFrame(dumpVar0);
                        break;
                    case 2:
                        var1.processFrame(dumpVar1);
                        break;
                    case 3:
                        var0.processFrame(dumpVar0);
                        var1.processFrame(dumpVar1);
                }
            }
            System.out.print("\n" + "ALIGNMENT: " + var0.frameCount + ", " + var1.frameCount + "\n");

            if (Thread.currentThread().isInterrupted()) {
                System.err.println("info: INTERRUPTED in doMatch()");
                return;
            }

            if (!batchMode) {
                var2.updateMatrix(true);
            }
        }

        if (var1.progressCallback != null) {
            var1.progressCallback.setFraction(1.0D);
        }

        if (!batchMode) {
            var2.updatePaths(false);
            var2.repaint();
        }

    }

    public static int processArgs(PerformanceMatcher var0, PerformanceMatcher var1, String[] var2) {
        for (int var3 = 0; var3 < var2.length; ++var3) {
            if (!silent) {
                System.err.println("args[" + var3 + "] = " + var2[var3]);
            }

            if (var2[var3].equals("-h")) {
                try {
                    ++var3;
                    var0.hopTime = var1.hopTime = Double.parseDouble(var2[var3]);
                } catch (RuntimeException var5) {
                    System.err.println(var5);
                }
            } else if (var2[var3].equals("-f")) {
                try {
                    ++var3;
                    var0.fftTime = var1.fftTime = Double.parseDouble(var2[var3]);
                } catch (RuntimeException var11) {
                    System.err.println(var11);
                }
            } else if (var2[var3].equals("-c")) {
                try {
                    ++var3;
                    var0.blockTime = var1.blockTime = Double.parseDouble(var2[var3]);
                } catch (RuntimeException var10) {
                    System.err.println(var10);
                }
            } else if (var2[var3].equals("-s")) {
                try {
                    ++var3;
                    var0.scale = var1.scale = Double.parseDouble(var2[var3]);
                } catch (RuntimeException var9) {
                    System.err.println(var9);
                }
            } else if (var2[var3].equals("-x")) {
                try {
                    ++var3;
                    var0.maxFrames = var1.maxFrames = Integer.parseInt(var2[var3]);
                } catch (RuntimeException var8) {
                    System.err.println(var8);
                }
            } else {
                String var10001;
                if (var2[var3].equals("-m1")) {
                    try {
                        ++var3;
                        var10001 = var2[var3];
                        ++var3;
                        var0.setMatchFile(var10001, Double.parseDouble(var2[var3]));
                    } catch (RuntimeException var7) {
                        System.err.println(var7);
                    }
                } else if (var2[var3].equals("-m2")) {
                    try {
                        ++var3;
                        var10001 = var2[var3];
                        ++var3;
                        var1.setMatchFile(var10001, Double.parseDouble(var2[var3]));
                    } catch (RuntimeException var6) {
                        System.err.println(var6);
                    }
                } else if (var2[var3].equals("-w1")) {
                    ++var3;
                    var0.setMatchFile(var2[var3], 0.0D, true);
                } else if (var2[var3].equals("-w2")) {
                    ++var3;
                    var1.setMatchFile(var2[var3], 0.0D, true);
                } else if (var2[var3].equals("-M1")) {
                    ++var3;
                    var0.setLabelFile(var2[var3]);
                    MatrixFrame.useSmoothPath = false;
                } else if (var2[var3].equals("-M2")) {
                    ++var3;
                    var1.setLabelFile(var2[var3]);
                    MatrixFrame.useSmoothPath = false;
                } else if (var2[var3].equals("-d")) {
                    var0.useSpectralDifference = var1.useSpectralDifference = true;
                } else if (var2[var3].equals("-D")) {
                    var0.useSpectralDifference = var1.useSpectralDifference = false;
                } else if (var2[var3].equals("-n1")) {
                    var0.normalise1 = var1.normalise1 = true;
                } else if (var2[var3].equals("-N1")) {
                    var0.normalise1 = var1.normalise1 = false;
                } else if (var2[var3].equals("-n2")) {
                    var0.normalise2 = var1.normalise2 = true;
                } else if (var2[var3].equals("-N2")) {
                    var0.normalise2 = var1.normalise2 = false;
                } else if (var2[var3].equals("-n3")) {
                    var0.normalise3 = var1.normalise3 = true;
                } else if (var2[var3].equals("-N3")) {
                    var0.normalise3 = var1.normalise3 = false;
                } else if (var2[var3].equals("-n4")) {
                    var0.normalise4 = var1.normalise4 = true;
                } else if (var2[var3].equals("-N4")) {
                    var0.normalise4 = var1.normalise4 = false;
                } else if (var2[var3].equals("-k1")) {
                    var0.normalise5 = true;
                    var1.normalise5 = false;
                } else if (var2[var3].equals("-k2")) {
                    var0.normalise5 = false;
                    var1.normalise5 = true;
                } else if (var2[var3].equals("--plot1")) {
                    var0.plot = new Plot();
                } else if (var2[var3].equals("--plot2")) {
                    var1.plot = new Plot();
                } else if (var2[var3].equals("--use-chroma-map")) {
                    var0.useChromaFrequencyMap = var1.useChromaFrequencyMap = true;
                } else if (var2[var3].equals("-b")) {
                    batchMode = true;
                    matrixVisible = false;
                    guiVisible = false;
                } else if (var2[var3].equals("-B")) {
                    batchMode = false;
                } else if (var2[var3].equals("-v")) {
                    matrixVisible = true;
                    batchMode = false;
                } else if (var2[var3].equals("-V")) {
                    matrixVisible = false;
                } else if (var2[var3].equals("-g")) {
                    guiVisible = true;
                    batchMode = false;
                } else if (var2[var3].equals("-G")) {
                    guiVisible = false;
                } else if (var2[var3].equals("-q")) {
                    silent = true;
                } else if (var2[var3].equals("-Q")) {
                    silent = false;
                } else if (var2[var3].equals("-a")) {
                    var0.audioOutputRequested = true;
                    var1.audioOutputRequested = false;
                } else if (var2[var3].equals("-A")) {
                    var1.audioOutputRequested = true;
                    var0.audioOutputRequested = false;
                } else if (var2[var3].equals("-r")) {
                    guiVisible = true;
                    batchMode = false;
                    ++var3;
                    GUI.loadFile = var2[var3];
                } else if (var2[var3].equals("-l")) {
                    var0.audioOutputRequested = true;
                    var1.audioOutputRequested = false;
                    var0.setLiveInput();
                } else if (var2[var3].equals("-L")) {
                    var1.audioOutputRequested = true;
                    var0.audioOutputRequested = false;
                    var1.setLiveInput();
                } else if (var2[var3].equals("-w")) {
                    var0.wormHandler = new WormHandler(var0);
                    var0.liveWorm = true;
                } else if (var2[var3].equals("-W")) {
                    var1.wormHandler = new WormHandler(var1);
                    var1.liveWorm = true;
                } else if (var2[var3].equals("-z")) {
                    if (var1.metadata == at.ofai.music.match.PerformanceMatcher.MetaType.WORM) {
                        var0.wormHandler = new WormHandler(var0);
                    }

                    ++var3;
                    var0.matchFileName = var2[var3];
                } else if (var2[var3].equals("-Z")) {
                    if (var0.metadata == at.ofai.music.match.PerformanceMatcher.MetaType.WORM) {
                        var1.wormHandler = new WormHandler(var1);
                    }

                    ++var3;
                    var1.matchFileName = var2[var3];
                } else if (var2[var3].equals("-ob")) {
                    ++var3;
                    var0.outputFileName = var2[var3];
                    var0.outputType = at.ofai.music.match.PerformanceMatcher.PathType.BACKWARD;
                } else if (var2[var3].equals("-of")) {
                    ++var3;
                    var0.outputFileName = var2[var3];
                    var0.outputType = at.ofai.music.match.PerformanceMatcher.PathType.FORWARD;
                } else if (var2[var3].equals("-os")) {
                    ++var3;
                    var0.outputFileName = var2[var3];
                    var0.outputType = at.ofai.music.match.PerformanceMatcher.PathType.SMOOTHED;
                } else if (var2[var3].equals("-rf1")) {
                    ++var3;
                    var0.referenceFrequency = Double.parseDouble(var2[var3]);
                } else if (var2[var3].equals("-F1")) { // dump features of first input
                    ++var3;
                    var0.featDumpFileName = var2[var3]; // example: "ref.feat.txt"
                } else if (var2[var3].equals("-F2")) { // dump features of second input
                    ++var3;
                    var1.featDumpFileName = var2[var3];
                } else {
                    if (!var2[var3].equals("-rf2")) {
                        return var3;
                    }

                    ++var3;
                    var1.referenceFrequency = Double.parseDouble(var2[var3]);
                }
            }
        }

        return var2.length;
    }

    public static void main(String[] var0) {
        PerformanceMatcher var1 = new PerformanceMatcher((PerformanceMatcher) null);
        PerformanceMatcher var2 = new PerformanceMatcher(var1);
        var1.setOtherMatcher(var2);
        int var3 = processArgs(var1, var2, var0);
        boolean headless = GraphicsEnvironment.isHeadless();
        boolean showGui = !headless && !batchMode;   // only show GUI if we have a display and we're not batch

        ScrollingMatrix var4 = new ScrollingMatrix(var1, var2);
        GUI var5 = null;
        if (showGui) {
            new MatrixFrame(var4, matrixVisible);
            var5 = new GUI(var1, var2, var4, guiVisible);
        }

        if (showGui && guiVisible && var3 != var0.length) {
            var5.addFiles(var0, var3);
        } else {
            if (var3 < var0.length && var1.pcmInputStream == null) {
                var1.setInputFile(var0[var3++]);
            }

            if (var3 < var0.length && var2.pcmInputStream == null) {
                var2.setInputFile(var0[var3++]);
            }
            if (var1.featDumpFileName != null) var1.dumpFeatures(var1.featDumpFileName);
            if (var2.featDumpFileName != null) var2.dumpFeatures(var2.featDumpFileName);

            if (var2.pcmInputStream != null) {
                doMatch(var1, var2, var4);
                var4.updatePaths(false);
                if (var1.metadata == var2.metadata
                        && (var1.metadata == at.ofai.music.match.PerformanceMatcher.MetaType.MATCH
                                || var1.metadata == at.ofai.music.match.PerformanceMatcher.MetaType.WORM)) {
                    var4.evaluatePaths();
                } else if (var1.metadata != at.ofai.music.match.PerformanceMatcher.MetaType.NONE
                        && var2.metadata == at.ofai.music.match.PerformanceMatcher.MetaType.NONE) {
                    if (var1.metadata == at.ofai.music.match.PerformanceMatcher.MetaType.LABEL) {
                        var2.writeLabelFile(var4);
                    } else if (var1.metadata == at.ofai.music.match.PerformanceMatcher.MetaType.MIDI) {
                        var2.writeMidiFile(var4);
                    } else if (var2.matchFileName != null) {
                        var4.wormHandler.write(new File(var2.matchFileName), false);
                    } else if (showGui && var5 != null) {
                        var5.saveWormFile();
                    }
                } else if (var1.outputFileName != null) {
                    switch ($SWITCH_TABLE$at$ofai$music$match$PerformanceMatcher$PathType()[var1.outputType
                            .ordinal()]) {
                        case 1:
                            var4.saveBackwardPath(var1.outputFileName);
                            break;
                        case 2:
                            var4.saveForwardPath(var1.outputFileName);
                            break;
                        case 3:
                            var4.saveSmoothedPath(var1.outputFileName);
                    }
                }
            }

            if (!silent) {
                System.err.println("Processed " + var1.frameCount + " and " + var2.frameCount + " frames of "
                        + var1.fftSize + " samples");
            }
        }

        if (batchMode) {
            System.exit(0);
        }
    }

    // $FF: synthetic method
    static int[] $SWITCH_TABLE$at$ofai$music$match$PerformanceMatcher$PathType() {
        int[] var10000 = $SWITCH_TABLE$at$ofai$music$match$PerformanceMatcher$PathType;
        if (var10000 != null) {
            return var10000;
        } else {
            int[] var0 = new int[at.ofai.music.match.PerformanceMatcher.PathType.values().length];

            try {
                var0[at.ofai.music.match.PerformanceMatcher.PathType.BACKWARD.ordinal()] = 1;
            } catch (NoSuchFieldError var3) {
            }

            try {
                var0[at.ofai.music.match.PerformanceMatcher.PathType.FORWARD.ordinal()] = 2;
            } catch (NoSuchFieldError var2) {
            }

            try {
                var0[at.ofai.music.match.PerformanceMatcher.PathType.SMOOTHED.ordinal()] = 3;
            } catch (NoSuchFieldError var1) {
            }

            $SWITCH_TABLE$at$ofai$music$match$PerformanceMatcher$PathType = var0;
            return var0;
        }
    }

    public static enum MetaType {
        NONE, MATCH, MIDI, LABEL, WORM;
    }

    public static enum PathType {
        BACKWARD, FORWARD, SMOOTHED;
    }

    public static class Onset {
        public double beat;
        public double time1;
        public double time2;
        public PerformanceMatcher parent;

        public Onset(PerformanceMatcher parent, double beat, double time1, double time2) {
            this.parent = parent;
            this.beat = beat;
            this.time1 = time1;
            this.time2 = time2;
        }
    }
}