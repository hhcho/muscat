package main

import (
	"encoding/binary"

	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"

	"log"
	"runtime/pprof"

	"github.com/hhcho/petchal/mhe"
	"github.com/ldsec/lattigo/v2/ckks"
)

func main() {

	command := os.Args[1]

	log.Println("Go MHE routine called:", command)

	switch command {

	case "setup":

		numClients, _ := strconv.Atoi(os.Args[2])
		localDirs := os.Args[3:]

		cpsList := mhe.NewCryptoParamsForNetwork(ckks.DefaultParams[ckks.PN14QP438], numClients, 0, 1)

		sharedKeyForClients := make([]byte, 32)
		rand.Read(sharedKeyForClients)

		for i, cps := range cpsList {

			mhe.SaveCryptoParamsAndRotKeys(i, localDirs[i], cps.Sk, cps.AggregateSk, cps.Pk, cps.Rlk, cps.RotKs)

			if i > 0 {

				err := mhe.WriteFullFile(localDirs[i]+"/shared_key.bin", sharedKeyForClients)
				if err != nil {
					log.Fatal("Error writing shared key")
				}

				if i == 1 {
					mhe.WriteFullFile(localDirs[i]+"/mask.bin", []byte{1})
				} else {
					mhe.WriteFullFile(localDirs[i]+"/mask.bin", []byte{0})
				}
			}

		}

	case "encrypt-vec":

		pid_path := os.Args[2]
		inFile := os.Args[3]
		outFile := os.Args[4]

		cps := mhe.NewCryptoParamsFromDiskPath(false, pid_path, 1)

		b, err := os.ReadFile(inFile)
		if err != nil {
			log.Fatal(err)
		}

		numBytes := 8

		vec := make([]float64, len(b)/numBytes)
		for i := range vec {
			// int64 version
			// vec[i] = float64(int64(binary.LittleEndian.Uint64(b[i*numBytes : (i+1)*numBytes])))

			// float64 version
			bits := binary.LittleEndian.Uint64(b[i*numBytes : (i+1)*numBytes])
			vec[i] = math.Float64frombits(bits)
		}

		vecEnc, _ := mhe.EncryptFloatVector(cps, vec)

		mhe.SaveCipherMatrixToFile(cps, mhe.CipherMatrix{vecEnc}, outFile)

	case "decrypt-test":

		pid_path := os.Args[2]
		inFile := os.Args[3]

		cps := mhe.NewCryptoParamsFromDiskPath(false, pid_path, 1)

		matDec, _ := mhe.LoadCipherMatrixFromFile(cps, inFile)

		out := mhe.DecryptFloatVector(cps, matDec[0], 5)

		log.Println(out)

	case "aggregate-cipher":

		pid_path := os.Args[2]
		outFile := os.Args[3]
		inFiles := os.Args[4:]

		cps := mhe.NewCryptoParamsFromDiskPath(true, pid_path, 1)

		var out mhe.CipherMatrix
		for _, f := range inFiles {
			matDec, _ := mhe.LoadCipherMatrixFromFile(cps, f)
			if out == nil {
				out = matDec
			} else {
				for i := range matDec {
					out[i] = mhe.CAdd(cps, out[i], matDec[i])
				}
			}
		}

		mhe.SaveCipherMatrixToFile(cps, out, outFile)

	case "decrypt-client-send":

		pidPath := os.Args[2]
		inFile := os.Args[3]
		outFile := os.Args[4]
		keyFile := os.Args[5]
		dimFile := os.Args[6]

		mhe.CollectiveDecryptClientSend(pidPath, inFile, outFile, keyFile, dimFile)

	case "decrypt-server":

		pidPath := os.Args[2]
		nr, _ := strconv.Atoi(os.Args[3])
		nc, _ := strconv.Atoi(os.Args[4])
		outFile := os.Args[5]
		inFiles := os.Args[6:]

		mhe.CollectiveDecryptServer(pidPath, nr, nc, outFile, inFiles)

	case "decrypt-client-receive":

		pidPath := os.Args[2]
		inFile := os.Args[3]
		serverFile := os.Args[4]
		keyFile := os.Args[5]
		outFile := os.Args[6]

		mhe.CollectiveDecryptClientReceive(pidPath, inFile, serverFile, keyFile, outFile)
	}

	// Run the Go profiler if enabled
	if os.Getenv("ENABLE_PPROF") == "true" {
		if err := writePProf(command); err != nil {
			log.Println("ERROR writing Profiler data: ", err)
		}
	}
}

const pprofType = "heap"

// Start the Go profiler if environment variable ENABLE_PPROF=true
func writePProf(command string) (err error) {
	// create output directory, if not present
	dir := "submission/pprof"
	if err = os.MkdirAll(dir, os.ModePerm); err != nil {
		return
	}

	// create the output file
	file := fmt.Sprintf("%s/%s_%s_%s.pprof", dir, command, pprofType, time.Now().Format(time.RFC3339))
	writer, err := os.Create(file)
	defer func() {
		writer.Close()
	}()
	if err != nil {
		return
	}

	// write the heap profile
	if err = pprof.Lookup(pprofType).WriteTo(writer, 0); err == nil {
		log.Printf("Wrote %s profile to %s", pprofType, file)
	}
	return
}
