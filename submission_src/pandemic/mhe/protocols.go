package mhe

import (
	"bufio"
	"encoding/binary"
	"io"
	"os"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/dckks"
	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/utils"
	"log"
)

// LoadSharedKeyAndRefresh loads the current state of the shared key and refreshes the key
func LoadSharedKeyAndRefresh(pidPath string) []byte {
	keyPath := pidPath + "/shared_key.bin"

	buf, _ := LoadFullFile(keyPath)

	prng, _ := utils.NewKeyedPRNG(buf)
	newKey := make([]byte, len(buf))

	prng.Read(buf)
	prng.Read(newKey)

	WriteFullFile(keyPath, newKey)

	return buf
}

// CollectiveDecryptClientSend performs the initial step of a collective decryption for a client
func CollectiveDecryptClientSend(pidPath, vectorToDecryptFile, outFile, keyFile, dimFile string) {
	cps := NewCryptoParamsFromDiskPath(false, pidPath, 1)
	parameters := cps.Params
	skShard := cps.Sk.Value
	cm, _ := LoadCipherMatrixFromFile(cps, vectorToDecryptFile)
	nr := len(cm)
	nc := len(cm[0])
	level := cm[0][0].Level()

	dckksContext := dckks.NewContext(cps.Params)

	zeroPoly := parameters.NewPolyQP()

	zeroPk := new(ckks.PublicKey)
	zeroPk.Value = [2]*ring.Poly{zeroPoly, zeroPoly}

	// Read mask flag
	flag, _ := LoadFullFile(pidPath + "/mask.bin")
	toMask := int(flag[0]) > 0

	log.Println("toMask:", toMask)

	// Sample two shared keys
	decToken := LoadSharedKeyAndRefresh(pidPath)
	pcksProtocol := dckks.NewPCKSProtocolDeterPRNG(parameters, 6.36, decToken)

	maskToken := LoadSharedKeyAndRefresh(pidPath)
	maskPrng, _ := utils.NewKeyedPRNG(maskToken)
	crpGen := ring.NewUniformSampler(maskPrng, dckksContext.RingQ)

	// Save keys for reuse
	WriteFullFile(keyFile+".1", decToken)
	WriteFullFile(keyFile+".2", maskToken)

	decShare := make([][]dckks.PCKSShare, nr)
	for i := range decShare {
		decShare[i] = make([]dckks.PCKSShare, nc)
		for j := range decShare[i] {
			decShare[i][j] = pcksProtocol.AllocateShares(level)
			pcksProtocol.GenShare(skShard, zeroPk, cm[i][j], decShare[i][j])
		}
	}

	file, err := os.Create(outFile)
	defer file.Close()
	if err != nil {
		log.Fatal(err)
	}

	writer := bufio.NewWriter(file)

	for r := range decShare {
		for c := range decShare[r] {
			for i := range decShare[r][c] {

				// Add mask
				if toMask {
					level := decShare[r][c][i].Level()
					mask := crpGen.ReadLvlNew(level)
					dckksContext.RingQ.AddLvl(level, decShare[r][c][i], mask, decShare[r][c][i])
				}

				decShareBytes, _ := decShare[r][c][i].MarshalBinary()
				AppendFullFile(writer, decShareBytes)
			}
		}
	}

	SaveFloatVectorToFile(dimFile, []float64{float64(nr), float64(nc)})
}

// CollectiveDecryptServer performs the server operation for a collective decryption
func CollectiveDecryptServer(pidPath string, nr, nc int, outFile string, inFiles []string) {
	cps := NewCryptoParamsFromDiskPath(true, pidPath, 1)

	dckksContext := dckks.NewContext(cps.Params)

	decShare := make([][]dckks.PCKSShare, nr)
	for r := range decShare {
		decShare[r] = make([]dckks.PCKSShare, nc)
		for c := range decShare[r] {
			decShare[r][c] = dckks.PCKSShare{nil, nil}
		}
	}

	for _, f := range inFiles {
		file, err := os.Open(f)
		defer file.Close()
		if err != nil {
			log.Fatal(err)
		}

		reader := bufio.NewReader(file)

		for r := range decShare {
			for c := range decShare[r] {
				for i := range decShare[r][c] {
					newPolyClient := new(ring.Poly)

					sbuf := make([]byte, 8)
					io.ReadFull(reader, sbuf)
					nBytes := binary.LittleEndian.Uint64(sbuf)

					buf := make([]byte, nBytes)
					io.ReadFull(reader, buf)

					newPolyClient.UnmarshalBinary(buf)

					if decShare[r][c][i] == nil {
						decShare[r][c][i] = newPolyClient
					} else {
						level := len(newPolyClient.Coeffs) - 1
						dckksContext.RingQ.AddLvl(level, decShare[r][c][i], newPolyClient, decShare[r][c][i])
					}
				}
			}
		}
	}

	// Save to file to send
	file, err := os.Create(outFile)
	defer file.Close()
	if err != nil {
		log.Fatal(err)
	}

	writer := bufio.NewWriter(file)

	for r := range decShare {
		for c := range decShare[r] {
			for i := range decShare[r][c] {
				decShareBytes, _ := decShare[r][c][i].MarshalBinary()
				AppendFullFile(writer, decShareBytes)
			}
		}
	}
}

// CollectiveDecryptClientReceive performs the final step of collective decryption at client
func CollectiveDecryptClientReceive(pidPath string, vectorToDecryptFile, server_polysPath, keyFile, outFile string) {
	cps := NewCryptoParamsFromDiskPath(false, pidPath, 1)

	parameters := cps.Params

	vectorToDecrypt, _ := LoadCipherMatrixFromFile(cps, vectorToDecryptFile)
	cm := vectorToDecrypt
	nr := len(cm)
	nc := len(cm[0])

	level := cm[0][0].Level()
	scale := cm[0][0].Scale()

	dckksContext := dckks.NewContext(cps.Params)

	decToken, _ := LoadFullFile(keyFile + ".1")
	pcksProtocol := dckks.NewPCKSProtocolDeterPRNG(parameters, 6.36, decToken)

	maskToken, _ := LoadFullFile(keyFile + ".2")
	maskPrng, _ := utils.NewKeyedPRNG(maskToken)
	crpGen := ring.NewUniformSampler(maskPrng, dckksContext.RingQ)

	decShare := make([][]dckks.PCKSShare, nr)
	for r := range decShare {
		decShare[r] = make([]dckks.PCKSShare, nc)
	}

	file, err := os.Open(server_polysPath)
	defer file.Close()
	if err != nil {
		log.Fatal(err)
	}

	reader := bufio.NewReader(file)

	for r := range decShare {
		for c := range decShare[r] {
			for i := range decShare[r][c] {
				newPolyServer := new(ring.Poly)

				sbuf := make([]byte, 8)
				io.ReadFull(reader, sbuf)
				nBytes := binary.LittleEndian.Uint64(sbuf)

				buf := make([]byte, nBytes)
				io.ReadFull(reader, buf)

				newPolyServer.UnmarshalBinary(buf)

				// Remove mask
				level := newPolyServer.Level()
				mask := crpGen.ReadLvlNew(level)
				dckksContext.RingQ.SubLvl(level, newPolyServer, mask, newPolyServer)

				decShare[r][c][i] = newPolyServer
			}
		}
	}

	pm := make(PlainVector, 0)
	for i := range cm {
		for j := range cm[i] {
			ciphertextSwitched := ckks.NewCiphertext(parameters, 1, level, scale)
			pcksProtocol.KeySwitch(decShare[i][j], cm[i][j], ciphertextSwitched)
			pm = append(pm, ciphertextSwitched.Plaintext())
		}
	}

	pmDecoded := DecodeFloatVector(cps, pm)

	SaveFloatVectorToFileBinary(outFile, pmDecoded)
}
