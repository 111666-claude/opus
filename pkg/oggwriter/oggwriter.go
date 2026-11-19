// SPDX-FileCopyrightText: 2026 The Pion community <https://pion.ly>
// SPDX-License-Identifier: MIT

// Package oggwriter implements an Ogg Opus container writer. It accepts
// pre-encoded Opus packets together with their duration and writes a
// standards-compliant Ogg Opus stream, per RFC 7845.
package oggwriter

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"sync"
	"unicode/utf8"
)

const (
	pageHeaderTypeContinuedPacket   = 0x01
	pageHeaderTypeBeginningOfStream = 0x02
	pageHeaderTypeEndOfStream       = 0x04
	pageHeaderSignature             = "OggS"
	pageVersion                     = 0
	pageHeaderSize                  = 27
	pageChecksumOffset              = 22
	pageSegmentCountOffset          = 26

	idPageSignature      = "OpusHead"
	commentPageSignature = "OpusTags"

	opusVersion      = 1
	opusSampleRate   = 48000
	opusHeadSize     = 19
	opusTagsConstant = 8 + 4 + 4 // "OpusTags", vendor length, user comment count

	defaultVendor = "pion"
	defaultSerial = 0

	// maxPageSegments is the largest lacing table an Ogg page can carry.
	maxPageSegments = 255
	// maxPageSegmentSize is the largest lacing value, which marks a segment
	// whose packet continues on the next page.
	maxPageSegmentSize = 255
	// maxPagePayload is the most packet data an Ogg page can hold.
	maxPagePayload = maxPageSegments * maxPageSegmentSize
	// maxSinglePagePacketSize is the largest packet that still fits on one
	// page once its terminating lacing value is counted.
	maxSinglePagePacketSize = maxPagePayload - 1
	// maxPacketPages bounds how many pages a single packet may span. It keeps
	// the amount of work per WritePacket bounded even if a caller hands over a
	// corrupt packet; a real Opus packet is at most 61,440 octets (RFC 7845
	// Section 6). 61,440 octets need 242 segments, so 2 pages would be enough;
	// the limit leaves room for oversized packets without unbounded work.
	maxPacketPages = 255
	// maxPacketSegments is the number of lacing values those pages can carry.
	maxPacketSegments = maxPacketPages * maxPageSegments
	// maxPacketSize is the largest packet that fits in maxPacketPages pages:
	// every lacing value but the last is a full 255-octet chunk, and the last
	// one is short so that the packet terminates there.
	maxPacketSize = (maxPacketSegments-1)*maxPageSegmentSize + maxPageSegmentSize - 1
	// maxPacketSamples is 120 ms at the 48 kHz granule rate, the longest
	// duration one Opus packet can represent.
	maxPacketSamples = opusSampleRate * 120 / 1000
	// maxVendorLength leaves room for the rest of the OpusTags packet on a
	// single page.
	maxVendorLength = maxSinglePagePacketSize - opusTagsConstant

	// noGranulePosition marks a page that is entirely spanned by a packet
	// that does not complete on it (RFC 7845 Section 4).
	noGranulePosition = ^uint64(0)
)

var (
	errNilWriter            = errors.New("writer is nil")
	errInvalidChannelCount  = errors.New("channel count must be 1 or 2")
	errInvalidVendor        = errors.New("vendor must be valid UTF-8")
	errVendorTooLong        = errors.New("vendor string is too long")
	errEmptyPacket          = errors.New("packet is empty")
	errInvalidSampleCount   = errors.New("sample count must be between 1 and 5760")
	errPacketTooLarge       = errors.New("packet exceeds the maximum page span")
	errSampleCountOverflow  = errors.New("total sample count overflows the granule position")
	errPageSequenceOverflow = errors.New("page sequence number overflows")
	errWriterClosed         = errors.New("writer is already closed")
	errWriteFailed          = errors.New("failed to write Ogg page")
)

// Option configures an OggWriter.
type Option func(*OggWriter) error

// WithPreSkip sets the pre-skip value written to the OpusHead header. It is
// the number of samples at 48 kHz that a player should discard from the
// decoder output when playback starts. The default is zero.
func WithPreSkip(preSkip uint16) Option {
	return func(writer *OggWriter) error {
		writer.preSkip = preSkip

		return nil
	}
}

// WithVendor sets the vendor string written to the OpusTags header. The
// default is "pion". The string must be valid UTF-8 and short enough to keep
// the OpusTags packet on a single page.
func WithVendor(vendor string) Option {
	return func(writer *OggWriter) error {
		if !utf8.ValidString(vendor) {
			return errInvalidVendor
		}
		if len(vendor) > maxVendorLength {
			return fmt.Errorf("%w: %d bytes, limit %d", errVendorTooLong, len(vendor), maxVendorLength)
		}
		writer.vendor = vendor

		return nil
	}
}

// WithSerial sets the logical bitstream serial number written to every page.
// The default is zero, which keeps the output deterministic for tests and
// single-stream files.
func WithSerial(serial uint32) Option {
	return func(writer *OggWriter) error {
		writer.serial = serial

		return nil
	}
}

// sealedPage is a finished page waiting to be written. The last page of a
// stream is held back so Close can set its end-of-stream flag without seeking
// or rewriting earlier output.
type sealedPage struct {
	lacing     []byte
	payload    []byte
	headerType uint8
	granulePos uint64
}

// OggWriter writes Opus packets into an Ogg Opus logical bitstream.
//
// The OpusHead and OpusTags pages are written by NewWith. Audio packets are
// framed into pages on WritePacket and the last page is completed by Close.
// The writer never buffers more than one page plus the write in progress, so
// long packets are streamed across pages instead of being copied whole.
type OggWriter struct {
	mu            sync.Mutex
	out           io.Writer
	checksumTable *[256]uint32

	preSkip  uint16
	vendor   string
	serial   uint32
	channels int
	seq      uint32
	granule  uint64

	closed bool
	err    error

	// Current audio page under construction.
	pageOpen       bool
	pageHeaderType uint8
	pageGranule    uint64
	lacing         []byte
	payload        []byte
	continuedNext  bool

	// pending is the most recently sealed page, written when the next page
	// starts or by Close. At most one page is ever held back, so buffering
	// stays bounded by one Ogg page regardless of packet size.
	pending *sealedPage
}

// NewWith returns an OggWriter that writes an Ogg Opus stream to out. channels
// must be 1 (mono) or 2 (stereo). The OpusHead and OpusTags header pages are
// written before NewWith returns.
func NewWith(out io.Writer, channels int, opts ...Option) (*OggWriter, error) {
	if out == nil {
		return nil, errNilWriter
	}
	if channels != 1 && channels != 2 {
		return nil, fmt.Errorf("%w: %d", errInvalidChannelCount, channels)
	}

	writer := &OggWriter{
		out:           out,
		checksumTable: generateChecksumTable(),
		vendor:        defaultVendor,
		serial:        defaultSerial,
		pageGranule:   noGranulePosition,
	}
	writer.channels = channels

	for _, opt := range opts {
		if err := opt(writer); err != nil {
			return nil, err
		}
	}

	if err := writer.writeHeaders(); err != nil {
		return nil, err
	}

	return writer, nil
}

// WritePacket appends one pre-encoded Opus packet to the stream. sampleCount
// is the number of samples per channel the packet represents at 48 kHz; it
// must be between 1 and 5760 (120 ms), and it becomes the increment of the
// page granule position.
//
// Passing a sample count smaller than the packet's decoded duration on the
// final packet trims the end of the stream, as allowed by RFC 7845
// Section 4.4. Packets are framed across as many pages as needed; the writer
// holds at most one page of data at a time.
func (w *OggWriter) WritePacket(packet []byte, sampleCount int) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if err := w.checkPacket(packet, sampleCount); err != nil {
		return err
	}

	remaining := packet
	for len(remaining) >= maxPageSegmentSize {
		if err := w.appendSegment(maxPageSegmentSize, remaining[:maxPageSegmentSize]); err != nil {
			return err
		}
		remaining = remaining[maxPageSegmentSize:]
	}

	if err := w.appendSegment(byte(len(remaining)), remaining); err != nil { //nolint:gosec // len(remaining) is below 255.
		return err
	}

	w.granule += uint64(sampleCount) //nolint:gosec // sampleCount was validated above.
	w.pageGranule = w.granule

	return nil
}

// checkPacket reports whether packet can be appended without touching the
// output. All validation happens here so a rejected packet writes nothing.
func (w *OggWriter) checkPacket(packet []byte, sampleCount int) error {
	if w.closed {
		return errWriterClosed
	}
	if w.err != nil {
		return w.err
	}
	if len(packet) == 0 {
		return errEmptyPacket
	}
	if sampleCount < 1 || sampleCount > maxPacketSamples {
		return fmt.Errorf("%w: %d", errInvalidSampleCount, sampleCount)
	}
	if len(packet) > maxPacketSize {
		return fmt.Errorf("%w: %d bytes, limit %d", errPacketTooLarge, len(packet), maxPacketSize)
	}
	// The all-ones granule value is reserved for pages without a completed
	// packet, so the largest usable granule is MaxUint64-1.
	if uint64(sampleCount) > math.MaxUint64-1-w.granule { //nolint:gosec // sampleCount was validated above.
		return fmt.Errorf("%w: %d + %d", errSampleCountOverflow, w.granule, sampleCount)
	}

	return nil
}

// Close finishes the logical stream. If audio was written, the final page is
// written with the end-of-stream flag set. A stream without audio still gets
// an empty end-of-stream page. Close does not close the underlying writer.
//
// Calling Close more than once returns errWriterClosed. A write error is
// remembered and returned by WritePacket and Close.
func (w *OggWriter) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.closed {
		return errWriterClosed
	}
	w.closed = true

	if w.err != nil {
		return w.err
	}

	if w.pageOpen {
		w.sealPage()
	}
	if w.pending != nil {
		page := w.pending
		w.pending = nil

		if err := w.writeSealedPage(page, true); err != nil {
			w.err = err

			return err
		}

		return nil
	}

	return w.writeEmptyEndOfStreamPage()
}

// writeHeaders writes the two mandatory header packets, each alone on its own
// page with a granule position of zero (RFC 7845 Section 4).
func (w *OggWriter) writeHeaders() error {
	if err := w.writeHeaderPage(w.buildIDHeader(), pageHeaderTypeBeginningOfStream); err != nil {
		return err
	}

	// WithVendor keeps the vendor short enough that OpusTags also fits on one
	// page, and the ID header is a fixed 19 bytes for family 0.
	return w.writeHeaderPage(w.buildCommentHeader(), 0)
}

func (w *OggWriter) writeHeaderPage(packet []byte, headerType uint8) error {
	if err := w.checkSequence(); err != nil {
		return err
	}

	lacing := packetLacing(packet)
	page := w.buildPage(headerType, 0, w.seq, lacing, packet)
	if err := w.writePage(page); err != nil {
		w.err = err

		return err
	}
	w.seq++

	return nil
}

func (w *OggWriter) buildIDHeader() []byte {
	header := make([]byte, opusHeadSize)
	copy(header, idPageSignature)
	header[8] = opusVersion
	header[9] = byte(w.channels) //nolint:gosec // channels is 1 or 2.
	binary.LittleEndian.PutUint16(header[10:], w.preSkip)
	binary.LittleEndian.PutUint32(header[12:], opusSampleRate)
	// Output gain is left at zero; channel mapping family 0 (mono/stereo)
	// needs no channel mapping table.
	return header
}

func (w *OggWriter) buildCommentHeader() []byte {
	header := make([]byte, opusTagsConstant+len(w.vendor))
	copy(header, commentPageSignature)
	binary.LittleEndian.PutUint32(header[8:], uint32(len(w.vendor))) //nolint:gosec // bounded by maxVendorLength.
	copy(header[12:], w.vendor)
	binary.LittleEndian.PutUint32(header[12+len(w.vendor):], 0)

	return header
}

// appendSegment adds one lacing value and its data to the current page,
// starting or finishing a page when needed. A page holds at most 255 lacing
// values, so a longer packet continues on the next page.
func (w *OggWriter) appendSegment(lacing byte, data []byte) error {
	if w.pageOpen && len(w.lacing) == maxPageSegments {
		w.sealPage()
	}
	if !w.pageOpen {
		if err := w.beginPage(); err != nil {
			return err
		}
	}

	w.lacing = append(w.lacing, lacing)
	w.payload = append(w.payload, data...)

	return nil
}

func (w *OggWriter) beginPage() error {
	if w.pending != nil {
		if err := w.flushPendingPage(); err != nil {
			return err
		}
	}

	w.pageOpen = true
	w.pageHeaderType = 0
	if w.continuedNext {
		w.pageHeaderType = pageHeaderTypeContinuedPacket
	}
	w.continuedNext = false
	w.pageGranule = noGranulePosition
	w.lacing = nil
	w.payload = nil

	return nil
}

func (w *OggWriter) sealPage() {
	w.continuedNext = len(w.lacing) > 0 && w.lacing[len(w.lacing)-1] == maxPageSegmentSize
	w.pending = &sealedPage{
		lacing:     w.lacing,
		payload:    w.payload,
		headerType: w.pageHeaderType,
		granulePos: w.pageGranule,
	}
	w.pageOpen = false
	w.lacing = nil
	w.payload = nil
}

func (w *OggWriter) flushPendingPage() error {
	if w.pending == nil {
		return nil
	}
	page := w.pending
	w.pending = nil

	if err := w.writeSealedPage(page, false); err != nil {
		w.err = err

		return err
	}

	return nil
}

func (w *OggWriter) writeSealedPage(page *sealedPage, endOfStream bool) error {
	if err := w.checkSequence(); err != nil {
		return err
	}

	headerType := page.headerType
	if endOfStream {
		headerType |= pageHeaderTypeEndOfStream
	}

	data := w.buildPage(headerType, page.granulePos, w.seq, page.lacing, page.payload)
	if err := w.writePage(data); err != nil {
		return err
	}
	w.seq++

	return nil
}

func (w *OggWriter) writeEmptyEndOfStreamPage() error {
	if err := w.checkSequence(); err != nil {
		return err
	}

	page := w.buildPage(pageHeaderTypeEndOfStream, w.granule, w.seq, nil, nil)
	if err := w.writePage(page); err != nil {
		w.err = err

		return err
	}
	w.seq++

	return nil
}

// checkSequence rejects a write that would wrap the 32-bit page sequence
// number, which Ogg cannot represent.
func (w *OggWriter) checkSequence() error {
	if w.seq == math.MaxUint32 {
		w.err = errPageSequenceOverflow

		return w.err
	}

	return nil
}

func (w *OggWriter) writePage(page []byte) error {
	written, err := w.out.Write(page)
	if err != nil {
		return fmt.Errorf("%w: %w", errWriteFailed, err)
	}
	if written != len(page) {
		return fmt.Errorf("%w: %w", errWriteFailed, io.ErrShortWrite)
	}

	return nil
}

// buildPage assembles one page and fills in its checksum. The page sequence
// number is not incremented here.
func (w *OggWriter) buildPage(headerType uint8, granulePos uint64, seq uint32, lacing, payload []byte) []byte {
	page := make([]byte, pageHeaderSize+len(lacing)+len(payload))

	copy(page, pageHeaderSignature)
	page[4] = pageVersion
	page[5] = headerType
	binary.LittleEndian.PutUint64(page[6:], granulePos)
	binary.LittleEndian.PutUint32(page[14:], w.serial)
	binary.LittleEndian.PutUint32(page[18:], seq)
	page[pageSegmentCountOffset] = byte(len(lacing)) //nolint:gosec // lacing is capped at maxPageSegments.
	copy(page[pageHeaderSize:], lacing)
	copy(page[pageHeaderSize+len(lacing):], payload)

	checksum := oggChecksum(w.checksumTable, page)
	binary.LittleEndian.PutUint32(page[pageChecksumOffset:], checksum)

	return page
}

// packetLacing returns the lacing values that encode one whole packet.
func packetLacing(packet []byte) []byte {
	lacing := make([]byte, 0, len(packet)/maxPageSegmentSize+1)
	remaining := len(packet)
	for remaining >= maxPageSegmentSize {
		lacing = append(lacing, maxPageSegmentSize)
		remaining -= maxPageSegmentSize
	}
	lacing = append(lacing, byte(remaining)) //nolint:gosec // remaining is below 255.

	return lacing
}

func generateChecksumTable() *[256]uint32 {
	var table [256]uint32
	const poly = 0x04c11db7

	for index := range uint32(256) {
		remainder := index << 24

		for range 8 {
			if (remainder & 0x80000000) != 0 {
				remainder = (remainder << 1) ^ poly
			} else {
				remainder <<= 1
			}
		}
		table[index] = remainder
	}

	return &table
}

// oggChecksum computes the Ogg page checksum with the checksum field treated
// as zero.
func oggChecksum(table *[256]uint32, page []byte) uint32 {
	var checksum uint32
	for index, value := range page {
		if index >= pageChecksumOffset && index < pageSegmentCountOffset {
			value = 0
		}
		checksum = (checksum << 8) ^ table[byte(checksum>>24)^value]
	}

	return checksum
}
