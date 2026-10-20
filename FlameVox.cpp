// ============================================================================
//  FlameVox - all-in-one vocal suite (VST3) built with JUCE 8
//  Chain: Auto-Tune -> Multiband Compressor -> Saturator -> Chorus
//         -> Spacey Ping-Pong Delay -> Reverb
// ============================================================================
#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_dsp/juce_dsp.h>

#include <array>
#include <cmath>
#include <vector>

// ----------------------------------------------------------------------------
//  Parameter IDs
// ----------------------------------------------------------------------------
namespace IDs
{
    static const juce::String tuneOn   = "tuneOn";
    static const juce::String key      = "key";
    static const juce::String scale    = "scale";
    static const juce::String retune   = "retune";
    static const juce::String low      = "low";
    static const juce::String mid      = "mid";
    static const juce::String high     = "high";
    static const juce::String chorus   = "chorus";
    static const juce::String saturate = "saturate";
    static const juce::String delay    = "delay";
    static const juce::String reverb   = "reverb";
}

static juce::AudioProcessorValueTreeState::ParameterLayout createLayout()
{
    using namespace juce;
    AudioProcessorValueTreeState::ParameterLayout layout;

    layout.add (std::make_unique<AudioParameterBool> (ParameterID { IDs::tuneOn, 1 }, "Tune On", true));
    layout.add (std::make_unique<AudioParameterChoice> (ParameterID { IDs::key, 1 }, "Key",
                    StringArray { "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" }, 0));
    layout.add (std::make_unique<AudioParameterChoice> (ParameterID { IDs::scale, 1 }, "Scale",
                    StringArray { "Major", "Minor", "Chromatic" }, 0));

    auto knob = [&] (const String& id, const String& name, float def)
    {
        layout.add (std::make_unique<AudioParameterFloat> (ParameterID { id, 1 }, name,
                        NormalisableRange<float> (0.0f, 1.0f, 0.001f), def));
    };

    knob (IDs::retune,   "Retune Speed", 0.7f);
    knob (IDs::low,      "Low Comp",     0.0f);
    knob (IDs::mid,      "Mid Comp",     0.0f);
    knob (IDs::high,     "High Comp",    0.0f);
    knob (IDs::chorus,   "Chorus",       0.0f);
    knob (IDs::saturate, "Saturator",    0.0f);
    knob (IDs::delay,    "Space Delay",  0.0f);
    knob (IDs::reverb,   "Reverb",       0.0f);
    return layout;
}

// ----------------------------------------------------------------------------
//  Auto-Tune pieces
// ----------------------------------------------------------------------------

// YIN pitch detector working on a ring buffer of the incoming mono signal
struct PitchDetector
{
    static constexpr int N       = 1024;   // integration window
    static constexpr int bufSize = 4096;   // ring size (power of two)
    static constexpr int mask    = bufSize - 1;

    void prepare()
    {
        ring.assign (bufSize, 0.0f);
        lin.assign (bufSize, 0.0f);
        dp.assign (2200, 1.0f);
        writeIdx = 0;
    }

    void push (float x) { ring[(size_t) writeIdx] = x; writeIdx = (writeIdx + 1) & mask; }

    // returns frequency in Hz, or 0 if unvoiced / silent
    float detect (double sr)
    {
        const int tauMax = juce::jmin ((int) (sr / 70.0), 2000);
        const int tauMin = juce::jmax (2, (int) (sr / 1000.0));
        const int need   = N + tauMax;

        for (int k = 0; k < need; ++k)
            lin[(size_t) k] = ring[(size_t) ((writeIdx - need + k) & mask)];

        float e = 0.0f;
        for (int k = tauMax; k < need; ++k)
            e += lin[(size_t) k] * lin[(size_t) k];
        if (std::sqrt (e / (float) N) < 0.004f)
            return 0.0f;

        const float* a = lin.data();
        float running = 0.0f;
        dp[0] = 1.0f;
        for (int tau = 1; tau <= tauMax; ++tau)
        {
            float sum = 0.0f;
            for (int j = 0; j < N; ++j)
            {
                const float d = a[j] - a[j + tau];
                sum += d * d;
            }
            running += sum;
            dp[(size_t) tau] = running > 0.0f ? sum * (float) tau / running : 1.0f;
        }

        int best = -1;
        for (int tau = tauMin; tau < tauMax; ++tau)
        {
            if (dp[(size_t) tau] < 0.18f)
            {
                while (tau + 1 < tauMax && dp[(size_t) tau + 1] < dp[(size_t) tau])
                    ++tau;
                best = tau;
                break;
            }
        }
        if (best < 2 || best >= tauMax)
            return 0.0f;

        const float x0 = dp[(size_t) best - 1], x1 = dp[(size_t) best], x2 = dp[(size_t) best + 1];
        const float denom = x0 - 2.0f * x1 + x2;
        const float shift = std::abs (denom) > 1.0e-9f ? 0.5f * (x0 - x2) / denom : 0.0f;
        const float f = (float) (sr / ((double) best + (double) shift));
        return (f > 70.0f && f < 1100.0f) ? f : 0.0f;
    }

    std::vector<float> ring, lin, dp;
    int writeIdx = 0;
};

// Dual-tap delay-line pitch shifter (Hann crossfaded), with a fixed-latency dry tap
struct DelayShifter
{
    void prepare (int windowSamples)
    {
        W = windowSamples;
        buf.assign ((size_t) (W * 2 + 8), 0.0f);
        pos = 0;
        phase = 0.0f;
    }

    float tap (float d) const
    {
        const int size = (int) buf.size();
        float rp = (float) pos - d;
        if (rp < 0.0f) rp += (float) size;
        int i0 = (int) rp;
        if (i0 >= size) i0 -= size;
        int i1 = i0 + 1;
        if (i1 >= size) i1 -= size;
        const float f = rp - (float) i0;
        return buf[(size_t) i0] * (1.0f - f) + buf[(size_t) i1] * f;
    }

    float process (float x, float ratio, float wet)
    {
        const int size = (int) buf.size();
        buf[(size_t) pos] = x;

        phase += (1.0f - ratio) / (float) W;
        phase -= std::floor (phase);
        float p2 = phase + 0.5f;
        p2 -= std::floor (p2);

        const float twoPi = juce::MathConstants<float>::twoPi;
        const float w1 = 0.5f - 0.5f * std::cos (twoPi * phase);
        const float w2 = 0.5f - 0.5f * std::cos (twoPi * p2);
        const float shifted = tap (phase * (float) W) * w1 + tap (p2 * (float) W) * w2;
        const float dry     = tap (0.5f * (float) W);

        pos = (pos + 1) % size;
        return dry + wet * (shifted - dry);
    }

    std::vector<float> buf;
    int W = 1024, pos = 0;
    float phase = 0.0f;
};

class AutoTune
{
public:
    void prepare (double sr)
    {
        sampleRate = sr;
        W = (int) (0.03 * sr);
        shifter[0].prepare (W);
        shifter[1].prepare (W);
        detector.prepare();
        hop = sr > 50000.0 ? 1024 : 512;
        hopCounter = 0;
        corr = corrTarget = 0.0f;
    }

    int getLatency() const { return W / 2; }

    static float snapMidi (float midi, int key, int scaleType)
    {
        if (scaleType >= 2)
            return std::round (midi);

        static const bool major[12] = { 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1 };
        static const bool minor[12] = { 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0 };
        const bool* mask = scaleType == 0 ? major : minor;

        const int base = (int) std::floor (midi);
        float best = std::round (midi), bestDist = 1.0e9f;
        for (int n = base - 2; n <= base + 3; ++n)
        {
            const int pc = (((n - key) % 12) + 12) % 12;
            if (mask[pc])
            {
                const float dist = std::abs (midi - (float) n);
                if (dist < bestDist) { bestDist = dist; best = (float) n; }
            }
        }
        return best;
    }

    void process (float* L, float* R, int n, bool on, int key, int scaleType, float retune)
    {
        const float tau   = 0.4f * std::pow (0.0025f, retune);          // 400 ms ... 1 ms
        const float coeff = 1.0f - std::exp (-1.0f / (tau * (float) sampleRate));

        for (int i = 0; i < n; ++i)
        {
            detector.push (0.5f * (L[i] + R[i]));

            if (++hopCounter >= hop)
            {
                hopCounter = 0;
                const float f = detector.detect (sampleRate);
                if (on && f > 0.0f)
                {
                    const float midi   = 69.0f + 12.0f * std::log2 (f / 440.0f);
                    const float target = snapMidi (midi, key, scaleType);
                    corrTarget = juce::jlimit (-2.0f, 2.0f, target - midi);
                }
                else
                    corrTarget = 0.0f;
            }

            corr += coeff * (corrTarget - corr);
            const float ratio = std::exp2 (corr / 12.0f);
            const float wet   = juce::jlimit (0.0f, 1.0f, (std::abs (corr) - 0.01f) / 0.06f);

            L[i] = shifter[0].process (L[i], ratio, wet);
            R[i] = shifter[1].process (R[i], ratio, wet);
        }
    }

private:
    double sampleRate = 44100.0;
    int W = 1323, hop = 512, hopCounter = 0;
    float corr = 0.0f, corrTarget = 0.0f;
    PitchDetector detector;
    DelayShifter shifter[2];
};

// ----------------------------------------------------------------------------
//  Spacey ping-pong delay (tempo-synced dotted 8th, dark feedback)
// ----------------------------------------------------------------------------
struct SpaceDelay
{
    void prepare (double sr)
    {
        sampleRate = sr;
        size = (int) (sr * 2.5) + 4;
        bufL.assign ((size_t) size, 0.0f);
        bufR.assign ((size_t) size, 0.0f);
        w = 0;
        lpL = lpR = 0.0f;
        time.reset (sr, 0.3);
        time.setCurrentAndTargetValue ((float) (0.375 * sr));
    }

    float read (const std::vector<float>& b, float rp) const
    {
        int i0 = (int) rp;
        if (i0 >= size) i0 -= size;
        int i1 = i0 + 1;
        if (i1 >= size) i1 -= size;
        const float f = rp - (float) i0;
        return b[(size_t) i0] * (1.0f - f) + b[(size_t) i1] * f;
    }

    void process (float* L, float* R, int n, float mix, float timeSamples)
    {
        time.setTargetValue (juce::jlimit (0.05f * (float) sampleRate, 2.0f * (float) sampleRate, timeSamples));
        const float fb      = 0.38f + 0.30f * mix;
        const float lpCoeff = 1.0f - std::exp (-juce::MathConstants<float>::twoPi * 3500.0f / (float) sampleRate);

        for (int i = 0; i < n; ++i)
        {
            const float t = time.getNextValue();
            float rp = (float) w - t;
            while (rp < 0.0f) rp += (float) size;

            const float dl = read (bufL, rp);
            const float dr = read (bufR, rp);
            lpL += lpCoeff * (dl - lpL);
            lpR += lpCoeff * (dr - lpR);

            const float in = 0.5f * (L[i] + R[i]);
            bufL[(size_t) w] = in + lpR * fb;     // ping-pong: L feeds from R...
            bufR[(size_t) w] = lpL * fb;          // ...and R feeds from L
            w = (w + 1) % size;

            L[i] += lpL * mix;
            R[i] += lpR * mix;
        }
    }

    std::vector<float> bufL, bufR;
    int size = 1, w = 0;
    float lpL = 0.0f, lpR = 0.0f;
    double sampleRate = 44100.0;
    juce::SmoothedValue<float> time;
};

// ----------------------------------------------------------------------------
//  Processor
// ----------------------------------------------------------------------------
class FlameVoxProcessor : public juce::AudioProcessor
{
public:
    FlameVoxProcessor()
        : AudioProcessor (BusesProperties()
                              .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
                              .withOutput ("Output", juce::AudioChannelSet::stereo(), true)),
          apvts (*this, nullptr, "STATE", createLayout())
    {
        pTuneOn   = apvts.getRawParameterValue (IDs::tuneOn);
        pKey      = apvts.getRawParameterValue (IDs::key);
        pScale    = apvts.getRawParameterValue (IDs::scale);
        pRetune   = apvts.getRawParameterValue (IDs::retune);
        pLow      = apvts.getRawParameterValue (IDs::low);
        pMid      = apvts.getRawParameterValue (IDs::mid);
        pHigh     = apvts.getRawParameterValue (IDs::high);
        pChorus   = apvts.getRawParameterValue (IDs::chorus);
        pSaturate = apvts.getRawParameterValue (IDs::saturate);
        pDelay    = apvts.getRawParameterValue (IDs::delay);
        pReverb   = apvts.getRawParameterValue (IDs::reverb);
    }

    const juce::String getName() const override { return "FlameVox"; }
    bool acceptsMidi() const override  { return false; }
    bool producesMidi() const override { return false; }
    double getTailLengthSeconds() const override { return 5.0; }
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}
    bool hasEditor() const override { return true; }
    juce::AudioProcessorEditor* createEditor() override;

    bool isBusesLayoutSupported (const BusesLayout& l) const override
    {
        return l.getMainOutputChannelSet() == juce::AudioChannelSet::stereo()
            && l.getMainInputChannelSet()  == juce::AudioChannelSet::stereo();
    }

    void prepareToPlay (double sr, int blockSize) override
    {
        sampleRate = sr;
        juce::dsp::ProcessSpec spec { sr, (juce::uint32) juce::jmax (blockSize, 64), 2 };

        autoTune.prepare (sr);
        setLatencySamples (autoTune.getLatency());

        xover1.prepare (spec);  xover1.setCutoffFrequency (200.0f);
        xover2.prepare (spec);  xover2.setCutoffFrequency (2500.0f);
        apLow.prepare (spec);   apLow.setCutoffFrequency (2500.0f);
        apLow.setType (juce::dsp::LinkwitzRileyFilterType::allpass);
        xover1.reset(); xover2.reset(); apLow.reset();

        for (int b = 0; b < 3; ++b)
        {
            bands[(size_t) b].setSize (2, juce::jmax (blockSize, 512));
            comps[(size_t) b].prepare (spec);
            comps[(size_t) b].setAttack (10.0f);
            comps[(size_t) b].setRelease (120.0f);
            prevMakeup[b] = 1.0f;
        }

        chorus.prepare (spec);
        chorus.setRate (0.9f);
        chorus.setCentreDelay (8.0f);
        chorus.setFeedback (0.0f);

        spaceDelay.prepare (sr);
        reverb.prepare (spec);
        reverb.reset();
        prevDrive = 0.0f;
    }

    void releaseResources() override {}

    void processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) override
    {
        juce::ScopedNoDenormals noDenormals;
        const int n = buffer.getNumSamples();
        if (buffer.getNumChannels() < 2 || n == 0)
            return;

        // 1) Auto-Tune
        autoTune.process (buffer.getWritePointer (0), buffer.getWritePointer (1), n,
                          pTuneOn->load() > 0.5f, (int) pKey->load(), (int) pScale->load(), pRetune->load());

        // 2) Multiband compressor
        processMultiband (buffer, n);

        // 3) Saturator
        processSaturator (buffer, n);

        // 4) Chorus
        {
            const float a = pChorus->load();
            chorus.setMix (a * 0.5f);
            chorus.setDepth (0.2f + 0.5f * a);
            juce::dsp::AudioBlock<float> block (buffer);
            juce::dsp::ProcessContextReplacing<float> ctx (block);
            chorus.process (ctx);
        }

        // 5) Spacey delay (dotted 8th from host tempo)
        {
            double bpm = 120.0;
            if (auto* ph = getPlayHead())
            {
                auto pos = ph->getPosition();
                if (pos.hasValue())
                {
                    auto b = pos->getBpm();
                    if (b.hasValue()) bpm = *b;
                }
            }
            bpm = juce::jlimit (40.0, 300.0, bpm);
            const float timeSamples = (float) (0.75 * 60.0 / bpm * sampleRate);
            spaceDelay.process (buffer.getWritePointer (0), buffer.getWritePointer (1), n,
                                pDelay->load(), timeSamples);
        }

        // 6) Reverb
        {
            const float a = pReverb->load();
            juce::Reverb::Parameters rp;
            rp.roomSize   = 0.88f;
            rp.damping    = 0.45f;
            rp.width      = 1.0f;
            rp.freezeMode = 0.0f;
            rp.wetLevel   = a * 0.45f;
            rp.dryLevel   = 1.0f - 0.35f * a;
            reverb.setParameters (rp);
            juce::dsp::AudioBlock<float> block (buffer);
            juce::dsp::ProcessContextReplacing<float> ctx (block);
            reverb.process (ctx);
        }
    }

    void getStateInformation (juce::MemoryBlock& dest) override
    {
        if (auto xml = apvts.copyState().createXml())
            copyXmlToBinary (*xml, dest);
    }

    void setStateInformation (const void* data, int size) override
    {
        if (auto xml = getXmlFromBinary (data, size))
            if (xml->hasTagName (apvts.state.getType()))
                apvts.replaceState (juce::ValueTree::fromXml (*xml));
    }

    juce::AudioProcessorValueTreeState apvts;

private:
    void processMultiband (juce::AudioBuffer<float>& buffer, int n)
    {
        const float amt[3] = { pLow->load(), pMid->load(), pHigh->load() };
        if (amt[0] < 0.001f && amt[1] < 0.001f && amt[2] < 0.001f)
            return;

        for (auto& b : bands)
            if (b.getNumSamples() < n)
                b.setSize (2, n, false, false, true);

        for (int ch = 0; ch < 2; ++ch)
        {
            const float* in = buffer.getReadPointer (ch);
            float* lo = bands[0].getWritePointer (ch);
            float* md = bands[1].getWritePointer (ch);
            float* hi = bands[2].getWritePointer (ch);

            for (int i = 0; i < n; ++i)
            {
                float l, rest, m, h;
                xover1.processSample (ch, in[i], l, rest);
                xover2.processSample (ch, rest, m, h);
                lo[i] = apLow.processSample (ch, l);   // keep low band phase-aligned
                md[i] = m;
                hi[i] = h;
            }
        }

        for (int b = 0; b < 3; ++b)
        {
            auto& band = bands[(size_t) b];
            const float a = amt[b];
            float makeup = 1.0f;

            if (a >= 0.001f)
            {
                const float threshold = -a * 30.0f;
                const float ratio     = 1.0f + a * 5.0f;
                comps[(size_t) b].setThreshold (threshold);
                comps[(size_t) b].setRatio (ratio);

                juce::dsp::AudioBlock<float> block (band);
                auto sub = block.getSubBlock (0, (size_t) n);
                juce::dsp::ProcessContextReplacing<float> ctx (sub);
                comps[(size_t) b].process (ctx);

                const float makeupDb = -threshold * (1.0f - 1.0f / ratio) * 0.5f;
                makeup = juce::Decibels::decibelsToGain (makeupDb);
            }

            band.applyGainRamp (0, n, prevMakeup[b], makeup);
            prevMakeup[b] = makeup;
        }

        for (int ch = 0; ch < 2; ++ch)
        {
            buffer.copyFrom (ch, 0, bands[0], ch, 0, n);
            buffer.addFrom  (ch, 0, bands[1], ch, 0, n);
            buffer.addFrom  (ch, 0, bands[2], ch, 0, n);
        }
    }

    void processSaturator (juce::AudioBuffer<float>& buffer, int n)
    {
        const float drive = pSaturate->load();
        const float d0 = prevDrive;
        prevDrive = drive;
        if (d0 < 1.0e-4f && drive < 1.0e-4f)
            return;

        float* ptr[2] = { buffer.getWritePointer (0), buffer.getWritePointer (1) };
        for (int i = 0; i < n; ++i)
        {
            const float d    = d0 + (drive - d0) * (float) (i + 1) / (float) n;
            const float k    = 1.0f + d * 9.0f;
            const float comp = 1.0f / std::pow (k, 0.4f);
            for (int ch = 0; ch < 2; ++ch)
            {
                const float x = ptr[ch][i];
                const float s = std::tanh (k * x) * comp;
                ptr[ch][i] = x + d * (s - x);
            }
        }
    }

    double sampleRate = 44100.0;
    AutoTune autoTune;

    juce::dsp::LinkwitzRileyFilter<float> xover1, xover2, apLow;
    std::array<juce::AudioBuffer<float>, 3> bands;
    std::array<juce::dsp::Compressor<float>, 3> comps;
    float prevMakeup[3] = { 1.0f, 1.0f, 1.0f };
    float prevDrive = 0.0f;

    juce::dsp::Chorus<float> chorus;
    SpaceDelay spaceDelay;
    juce::dsp::Reverb reverb;

    std::atomic<float> *pTuneOn = nullptr, *pKey = nullptr, *pScale = nullptr, *pRetune = nullptr,
                       *pLow = nullptr, *pMid = nullptr, *pHigh = nullptr, *pChorus = nullptr,
                       *pSaturate = nullptr, *pDelay = nullptr, *pReverb = nullptr;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (FlameVoxProcessor)
};

// ----------------------------------------------------------------------------
//  GUI: black background, animated white flames, white-on-black controls
// ----------------------------------------------------------------------------
struct FlameLookAndFeel : public juce::LookAndFeel_V4
{
    FlameLookAndFeel()
    {
        const auto white = juce::Colours::white;
        setColour (juce::ComboBox::backgroundColourId, juce::Colours::black);
        setColour (juce::ComboBox::textColourId, white);
        setColour (juce::ComboBox::outlineColourId, white.withAlpha (0.6f));
        setColour (juce::ComboBox::arrowColourId, white);
        setColour (juce::ComboBox::buttonColourId, juce::Colours::black);
        setColour (juce::PopupMenu::backgroundColourId, juce::Colour (0xff0a0a0a));
        setColour (juce::PopupMenu::textColourId, white);
        setColour (juce::PopupMenu::highlightedBackgroundColourId, white);
        setColour (juce::PopupMenu::highlightedTextColourId, juce::Colours::black);
        setColour (juce::Label::textColourId, white);
        setColour (juce::Slider::textBoxTextColourId, white);
        setColour (juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
        setColour (juce::Slider::textBoxBackgroundColourId, juce::Colours::transparentBlack);
        setColour (juce::ToggleButton::textColourId, white);
        setColour (juce::ToggleButton::tickColourId, white);
        setColour (juce::ToggleButton::tickDisabledColourId, white.withAlpha (0.4f));
    }

    void drawRotarySlider (juce::Graphics& g, int x, int y, int w, int h, float pos,
                           float startAngle, float endAngle, juce::Slider&) override
    {
        auto bounds = juce::Rectangle<float> ((float) x, (float) y, (float) w, (float) h).reduced (6.0f);
        const float r = juce::jmin (bounds.getWidth(), bounds.getHeight()) * 0.5f;
        const auto c = bounds.getCentre();
        const float angle = startAngle + pos * (endAngle - startAngle);
        const float arcR = r - 4.0f;

        g.setColour (juce::Colours::black.withAlpha (0.8f));
        g.fillEllipse (c.x - r, c.y - r, 2.0f * r, 2.0f * r);
        g.setColour (juce::Colours::white.withAlpha (0.25f));
        g.drawEllipse (c.x - r, c.y - r, 2.0f * r, 2.0f * r, 1.5f);

        juce::Path track;
        track.addCentredArc (c.x, c.y, arcR, arcR, 0.0f, startAngle, endAngle, true);
        g.setColour (juce::Colours::white.withAlpha (0.15f));
        g.strokePath (track, juce::PathStrokeType (3.0f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

        juce::Path val;
        val.addCentredArc (c.x, c.y, arcR, arcR, 0.0f, startAngle, angle, true);
        g.setColour (juce::Colours::white);
        g.strokePath (val, juce::PathStrokeType (3.0f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

        juce::Path p;
        p.addRoundedRectangle (-1.5f, -arcR + 3.0f, 3.0f, arcR * 0.5f, 1.5f);
        p.applyTransform (juce::AffineTransform::rotation (angle).translated (c.x, c.y));
        g.fillPath (p);
    }
};

struct Knob : public juce::Component
{
    Knob (juce::AudioProcessorValueTreeState& state, const juce::String& id, const juce::String& text)
    {
        slider.setRotaryParameters (juce::MathConstants<float>::pi * 1.2f,
                                    juce::MathConstants<float>::pi * 2.8f, true);
        slider.setTextBoxStyle (juce::Slider::TextBoxBelow, false, 70, 18);
        addAndMakeVisible (slider);

        label.setText (text, juce::dontSendNotification);
        label.setJustificationType (juce::Justification::centred);
        label.setFont (juce::Font (juce::FontOptions (13.0f, juce::Font::bold)));
        addAndMakeVisible (label);

        attachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment> (state, id, slider);
        slider.textFromValueFunction = [] (double v) { return juce::String (juce::roundToInt (v * 100.0)) + "%"; };
        slider.updateText();
    }

    void resized() override
    {
        auto b = getLocalBounds();
        label.setBounds (b.removeFromTop (18));
        slider.setBounds (b);
    }

    juce::Slider slider { juce::Slider::RotaryHorizontalVerticalDrag, juce::Slider::TextBoxBelow };
    juce::Label label;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> attachment;
};

class FlameVoxEditor : public juce::AudioProcessorEditor, private juce::Timer
{
public:
    explicit FlameVoxEditor (FlameVoxProcessor& p)
        : juce::AudioProcessorEditor (&p),
          retune (p.apvts, IDs::retune, "RETUNE SPEED"),
          low (p.apvts, IDs::low, "LOW  (<200 Hz)"),
          mid (p.apvts, IDs::mid, "MID"),
          high (p.apvts, IDs::high, "HIGH  (>2.5 kHz)"),
          chorus (p.apvts, IDs::chorus, "CHORUS"),
          saturate (p.apvts, IDs::saturate, "SATURATOR"),
          delay (p.apvts, IDs::delay, "SPACE DELAY"),
          reverb (p.apvts, IDs::reverb, "REVERB")
    {
        setLookAndFeel (&laf);
        setSize (900, 680);

        keyBox.addItemList (juce::StringArray { "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" }, 1);
        scaleBox.addItemList (juce::StringArray { "Major", "Minor", "Chromatic" }, 1);
        tuneButton.setButtonText ("AUTO-TUNE ON");

        for (auto* l : { &keyLabel, &scaleLabel })
        {
            l->setJustificationType (juce::Justification::centredLeft);
            l->setFont (juce::Font (juce::FontOptions (13.0f, juce::Font::bold)));
            addAndMakeVisible (l);
        }
        keyLabel.setText ("KEY", juce::dontSendNotification);
        scaleLabel.setText ("SCALE", juce::dontSendNotification);

        addAndMakeVisible (keyBox);
        addAndMakeVisible (scaleBox);
        addAndMakeVisible (tuneButton);
        for (auto* k : { &retune, &low, &mid, &high, &chorus, &saturate, &delay, &reverb })
            addAndMakeVisible (k);

        keyAtt   = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment> (p.apvts, IDs::key, keyBox);
        scaleAtt = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment> (p.apvts, IDs::scale, scaleBox);
        tuneAtt  = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment> (p.apvts, IDs::tuneOn, tuneButton);

        startTimerHz (30);
    }

    ~FlameVoxEditor() override
    {
        stopTimer();
        setLookAndFeel (nullptr);
    }

    void resized() override
    {
        auto area = getLocalBounds().reduced (20);
        area.removeFromTop (44);
        tuneArea = area.removeFromTop (170);  area.removeFromTop (10);
        compArea = area.removeFromTop (170);  area.removeFromTop (10);
        fxArea   = area.removeFromTop (170);

        // Auto-tune row
        {
            auto inner = tuneArea.reduced (10);
            inner.removeFromTop (22);
            retune.setBounds (inner.removeFromRight (170));
            auto placeCombo = [] (juce::Label& l, juce::ComboBox& c, juce::Rectangle<int> r)
            {
                r = r.withTrimmedTop (22);
                l.setBounds (r.removeFromTop (20));
                c.setBounds (r.removeFromTop (34));
            };
            placeCombo (keyLabel, keyBox, inner.removeFromLeft (210).reduced (10, 0));
            placeCombo (scaleLabel, scaleBox, inner.removeFromLeft (210).reduced (10, 0));
            tuneButton.setBounds (inner.removeFromLeft (200).withTrimmedTop (64).removeFromTop (34).reduced (10, 0));
        }

        layoutRow (compArea, { &low, &mid, &high });
        layoutRow (fxArea, { &chorus, &saturate, &delay, &reverb });
    }

    void paint (juce::Graphics& g) override
    {
        g.fillAll (juce::Colours::black);
        drawFlames (g);

        g.setColour (juce::Colours::white);
        g.setFont (juce::Font (juce::FontOptions (30.0f, juce::Font::bold)));
        g.drawText ("FLAMEVOX", 0, 12, getWidth(), 40, juce::Justification::centred);

        drawPanel (g, tuneArea, "AUTO-TUNE");
        drawPanel (g, compArea, "MULTIBAND COMPRESSOR");
        drawPanel (g, fxArea,   "EFFECTS");
    }

private:
    void timerCallback() override
    {
        time += 1.0f / 30.0f;
        repaint();
    }

    void layoutRow (juce::Rectangle<int> panel, std::initializer_list<Knob*> knobs)
    {
        auto inner = panel.reduced (10);
        inner.removeFromTop (22);
        const int cnt = (int) knobs.size();
        const int cellW = inner.getWidth() / cnt;
        for (auto* k : knobs)
        {
            auto cell = inner.removeFromLeft (cellW);
            k->setBounds (cell.withSizeKeepingCentre (150, cell.getHeight()));
        }
    }

    void drawPanel (juce::Graphics& g, juce::Rectangle<int> r, const juce::String& title)
    {
        g.setColour (juce::Colours::black.withAlpha (0.72f));
        g.fillRoundedRectangle (r.toFloat(), 10.0f);
        g.setColour (juce::Colours::white.withAlpha (0.35f));
        g.drawRoundedRectangle (r.toFloat().reduced (0.5f), 10.0f, 1.2f);
        g.setColour (juce::Colours::white);
        g.setFont (juce::Font (juce::FontOptions (14.0f, juce::Font::bold)));
        g.drawText (title, r.getX() + 16, r.getY() + 8, r.getWidth() - 32, 20, juce::Justification::centredLeft);
    }

    static float rnd (int k)
    {
        const float s = std::sin ((float) k * 12.9898f) * 43758.5453f;
        return s - std::floor (s);
    }

    void drawFlames (juce::Graphics& g)
    {
        const float W = (float) getWidth(), H = (float) getHeight();
        const int count = 18;
        const float twoPi = juce::MathConstants<float>::twoPi;

        for (int layer = 0; layer < 3; ++layer)
        {
            const float scale = 1.0f - 0.28f * (float) layer;
            const float alpha = 0.10f + 0.16f * (float) layer;

            for (int i = 0; i < count; ++i)
            {
                const int s = i * 7 + layer * 31;
                const float x      = ((float) i + 0.5f) * W / (float) count + (rnd (s + 1) - 0.5f) * 30.0f;
                const float phase1 = rnd (s + 2) * twoPi;
                const float phase2 = rnd (s + 3) * twoPi;
                const float baseH  = H * (0.30f + 0.32f * rnd (s + 4)) * scale;
                const float flick  = 0.75f + 0.25f * std::sin (time * (2.5f + 3.0f * rnd (s + 5)) + phase1)
                                           + 0.12f * std::sin (time * (6.0f + 4.0f * rnd (s + 6)) + phase2);
                const float h      = baseH * flick;
                const float halfW  = (W / (float) count) * (0.75f + 0.5f * rnd (s + 7)) * (0.95f - 0.2f * (float) layer);
                const float sway   = std::sin (time * (1.2f + rnd (s + 8)) + phase1) * halfW * 0.8f;
                const float tipX   = x + sway;
                const float tipY   = H - h;

                juce::Path p;
                p.startNewSubPath (x - halfW, H);
                p.cubicTo (x - halfW * 1.1f, H - h * 0.40f, tipX - halfW * 0.25f, H - h * 0.72f, tipX, tipY);
                p.cubicTo (tipX + halfW * 0.25f, H - h * 0.68f, x + halfW * 1.1f, H - h * 0.35f, x + halfW, H);
                p.closeSubPath();

                juce::ColourGradient grad (juce::Colours::white.withAlpha (alpha + 0.12f), x, H,
                                           juce::Colours::white.withAlpha (0.0f), tipX, tipY, false);
                g.setGradientFill (grad);
                g.fillPath (p);

                g.setColour (juce::Colours::white.withAlpha (alpha * 0.9f));
                g.strokePath (p, juce::PathStrokeType (1.0f));
            }
        }
    }

    FlameLookAndFeel laf;

    Knob retune, low, mid, high, chorus, saturate, delay, reverb;
    juce::ComboBox keyBox, scaleBox;
    juce::Label keyLabel, scaleLabel;
    juce::ToggleButton tuneButton;

    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> keyAtt, scaleAtt;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> tuneAtt;

    juce::Rectangle<int> tuneArea, compArea, fxArea;
    float time = 0.0f;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (FlameVoxEditor)
};

juce::AudioProcessorEditor* FlameVoxProcessor::createEditor()
{
    return new FlameVoxEditor (*this);
}

// Plugin entry point
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new FlameVoxProcessor();
}
