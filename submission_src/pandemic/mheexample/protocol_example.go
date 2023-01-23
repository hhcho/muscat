package mheexample

import (
	"crypto/rand"
	"encoding/binary"
	"fmt"
	"github.com/hhcho/petchal/crypto"
	"github.com/hhcho/petchal/mpc"
	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/dckks"
	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/utils"
	"go.dedis.ch/onet/v3/log"
	"strconv"
	"time"
)

type ExampleProtocolInfo struct {
	Prot *ProtocolInfo
	Data []float64
}

func InitializeExampleProtocol(pid int, configFolder string) (exampleProt *ExampleProtocolInfo) {
	prot := InitializeProtocol(pid, configFolder)

	data := []float64{1.0, 2.0, 3.0}

	return &ExampleProtocolInfo{
		Data: data,
		Prot: prot,
	}
}

func (pi *ExampleProtocolInfo) ExampleProtocol() {

	// node ID and cryptoparams
	pid := pi.Prot.MpcObj[0].GetPid()
	cps := pi.Prot.Cps
	serverPIDNet := pi.Prot.MpcObj.GetNetworks()[0]

	fmt.Println(pid, " is ready")
	if pid > 0 {

		// dummy vectors for now, with 2 times 64 features
		var modelFeatures crypto.CipherVector
		//var featuresSize int //, featuresEncryptedSize int
		features := make([]float64, 64)
		for i := range features {
			features[i] = float64(1)
		}
		features = append(features, features...)
		modelFeatures, _ = crypto.EncryptFloatVector(cps, features)
		log.LLvl1("ENCRYPTED")
		if pid == 1 {
			crypto.SaveCipherMatrixToFile(cps, crypto.CipherMatrix{modelFeatures}, "test_encrypted_1")
			crypto.SaveCipherMatrixToFile(cps, crypto.CipherMatrix{modelFeatures}, "test_decryption")
		}

		log.LLvl1("SAVED TOF ILE")
		modelFeatures[0] = crypto.RotateAndAdd(cps, modelFeatures[0], 8)

		log.LLvl1("ICI ", crypto.DecryptFloatVector(cps, modelFeatures, 10))

		// TEST AGGREGATION
		dummyVector := []float64{1, 2, 3, 4, 5, 6}
		if pid == 2 {
			crypto.SaveFloatVectorToFile("test", dummyVector)
			ClientEncryptVector(pid, "test", 6)
			// test
			vectorRead, _ := crypto.LoadCipherMatrixFromFile(cps, "test_encrypted_1")
			log.LLvl1("CLIENT ", crypto.DecryptFloatVector(cps, vectorRead[0], 10))
		} else if pid == 1 {
			ServerAggregation("test_encrypted_", "result")
		}

		// TEST BOOTSTRAPPING *************************************************************************************
		// both pids play clients, pid1 is also server

		if pid == 1 {
			CollectiveBootstrapClientSend(pid, "test_encrypted_1", "boot", 2)
		} else if pid == 2 {
			CollectiveBootstrapClientSend(pid, "test_encrypted_1", "boot", 2)
		}

		// JUST FOR SYNCHRO
		if pid == 1 {
			matrixRead, _ := crypto.LoadCipherMatrixFromFile(cps, "result")
			plain := serverPIDNet.CollectiveDecryptVec(cps, matrixRead[0], pid)
			log.LLvl1(crypto.DecodeFloatVector(cps, plain)[:100])
		} else if pid > 0 {
			serverPIDNet.CollectiveDecryptVec(cps, nil, 1)
		}

		if pid == 1 { // plays the server role
			CollectiveBootstrapServer("boot", "bootstrapped")
		}

		// JUST FOR SYNCHRO
		if pid == 1 {
			matrixRead, _ := crypto.LoadCipherMatrixFromFile(cps, "result")
			plain := serverPIDNet.CollectiveDecryptVec(cps, matrixRead[0], pid)
			log.LLvl1(crypto.DecodeFloatVector(cps, plain)[:100])
		} else if pid > 0 {
			serverPIDNet.CollectiveDecryptVec(cps, nil, 1)
		}

		if pid == 1 {
			CollectiveBootstrapClientReceive(pid, "test_encrypted_1", "boot", "bootstrapped", "bootres")
			matrixRead, _ := crypto.LoadCipherMatrixFromFile(cps, "bootres")
			log.LLvl1("Bootstrap ", crypto.DecryptFloatVector(cps, matrixRead[0], 10))
		} else if pid == 2 {
			CollectiveBootstrapClientReceive(pid, "test_encrypted_1", "boot", "bootstrapped", "bootres_2")
			matrixReadNew, _ := crypto.LoadCipherMatrixFromFile(cps, "bootres_2")
			log.LLvl1("Bootstrap ", crypto.DecryptFloatVector(cps, matrixReadNew[0], 10))
		}

		// TEST DECRYPTION *************************************************************************************
		if pid == 1 {
			CollectiveDecryptClientSend(pid, "test_decryption", "dec")
		} else if pid == 2 {
			CollectiveDecryptClientSend(pid, "test_decryption", "dec")
		}

		// JUST FOR SYNCHRO
		if pid == 1 {
			matrixRead, _ := crypto.LoadCipherMatrixFromFile(cps, "result")
			plain := serverPIDNet.CollectiveDecryptVec(cps, matrixRead[0], pid)
			log.LLvl1(crypto.DecodeFloatVector(cps, plain)[:100])
		} else if pid > 0 {
			serverPIDNet.CollectiveDecryptVec(cps, nil, 1)
		}

		if pid == 1 { // plays the server role
			CollectiveDecryptServer("test_decryption", "dec", "decrypted")
		}

		// JUST FOR SYNCHRO
		if pid == 1 {
			matrixRead, _ := crypto.LoadCipherMatrixFromFile(cps, "result")
			plain := serverPIDNet.CollectiveDecryptVec(cps, matrixRead[0], pid)
			log.LLvl1(crypto.DecodeFloatVector(cps, plain)[:100])
		} else if pid > 0 {
			serverPIDNet.CollectiveDecryptVec(cps, nil, 1)
		}

		if pid == 1 {
			CollectiveDecryptClientReceive(pid, "test_decryption", "decrypted", "decryptedRes")
			matrixRead := crypto.LoadFloatVectorFromFile("decryptedRes", 10)
			log.LLvl1("decryption ", matrixRead)
		} else if pid == 2 {
			CollectiveDecryptClientReceive(pid, "test_decryption", "decrypted", "decryptedRes_2")
			matrixReadNew := crypto.LoadFloatVectorFromFile("decryptedRes_2", 10)
			log.LLvl1("decryption ", matrixReadNew)
		}

		if pid == 1 {
			matrixRead, _ := crypto.LoadCipherMatrixFromFile(cps, "result")
			plain := serverPIDNet.CollectiveDecryptVec(cps, matrixRead[0], pid)
			log.LLvl1(crypto.DecodeFloatVector(cps, plain)[:100])
		} else if pid > 0 {
			serverPIDNet.CollectiveDecryptVec(cps, nil, 1)
		}
	}

}

func ClientEncryptVector(pid int, vectorfileName string, vectorSize int) {
	cps := crypto.NewCryptoParamsFromDisk(false, pid, 1)
	vectorToAggr := crypto.LoadFloatVectorFromFile(vectorfileName, vectorSize)
	vectorToAggrEncr, _ := crypto.EncryptFloatVector(cps, vectorToAggr)

	// test decryption
	log.LLvl1("CLIENT ", crypto.DecryptFloatVector(cps, vectorToAggrEncr, 10))

	crypto.SaveCipherMatrixToFile(cps, crypto.CipherMatrix{vectorToAggrEncr}, vectorfileName+"_encrypted_"+strconv.Itoa(pid))

}

func ServerAggregation(vectorsToAggregateName string, out_path string) {
	cps := crypto.NewCryptoParamsFromDisk(true, 1, 1)

	fileIndex := 1
	var matricesToAggregate []crypto.CipherMatrix
	for {
		readMatrix, err := crypto.LoadCipherMatrixFromFile(cps, vectorsToAggregateName+strconv.Itoa(fileIndex))
		if err != nil {
			break
		}
		matricesToAggregate = append(matricesToAggregate, readMatrix)
		fileIndex++
	}

	aggregatedMatrix := crypto.AggregateMat(cps, matricesToAggregate)

	crypto.SaveCipherMatrixToFile(cps, aggregatedMatrix, out_path)
}

// REF CollectiveBootstrapMat
func CollectiveBootstrapClientSend(pid int, vectorToBootstrapFile string, out_path string, nbrOfClients int) {
	cps := crypto.NewCryptoParamsFromDisk(false, pid, 1)
	parameters := cps.Params
	skShard := cps.Sk.Value
	vectorToBootstrap, _ := crypto.LoadCipherMatrixFromFile(cps, vectorToBootstrapFile)
	cm := vectorToBootstrap

	dckksContext := dckks.NewContext(parameters)

	// TODO how to better do this crpGen := netObj.GetCRPGen()
	lattigoPRNG, _ := utils.NewKeyedPRNG([]byte{'l', 'a', 't', 't', 'i', 'g', 'o'})

	crpGen := ring.NewUniformSampler(lattigoPRNG, dckksContext.RingQP)

	cm, levelStart := crypto.FlattenLevels(cps, cm)
	log.LLvl1(time.Now().Format(time.RFC3339), "Bootstrap: dimensions", len(cm), "x", len(cm[0]), "input level", levelStart)

	// save key for reuse
	// TODO put party id
	token := make([]byte, 32)
	//rand.Read(token)
	crypto.WriteFullFile("token_bootstrapping_1", token)
	token2 := make([]byte, 32)
	//rand.Read(token2)
	crypto.WriteFullFile("token_bootstrapping_2", token2)
	refProtocol := dckks.NewRefreshProtocolDeterPRNGs(parameters, token, token2)

	refSharesDecrypt := make([][]ring.Poly, len(cm))
	crps := make([][]ring.Poly, len(cm))
	refSharesRecrypt := make([][]ring.Poly, len(cm))

	for i := range cm {
		refSharesDecrypt[i] = make([]ring.Poly, len(cm[0]))
		crps[i] = make([]ring.Poly, len(cm[0]))
		refSharesRecrypt[i] = make([]ring.Poly, len(cm[0]))

		for j := range cm[i] {
			refShare1, refShare2 := refProtocol.AllocateShares(levelStart)

			refSharesDecrypt[i][j] = *refShare1
			refSharesRecrypt[i][j] = *refShare2
			crps[i][j] = *crpGen.ReadNew()

			refProtocol.GenShares(skShard, levelStart, nbrOfClients-1, cm[i][j], parameters.Scale(),
				&(crps[i][j]), &(refSharesDecrypt[i][j]), &(refSharesRecrypt[i][j]))

		}
	}
	// save local crps
	sizesCrps, bytesCrps := mpc.MarshalPolyMat(crps)
	crypto.WriteFullFile(out_path+"_"+strconv.Itoa(pid)+"_crpsSizes", sizesCrps)
	crypto.WriteFullFile(out_path+"_"+strconv.Itoa(pid)+"_crps", bytesCrps)

	//refAgg1
	//contextQ := dckksContext.RingQ
	//shareOut := make([][]ring.Poly, len(refSharesDecrypt))
	// SendPolyMat (both in one)
	sizesDec, bytesDec := mpc.MarshalPolyMat(refSharesDecrypt)
	bs := make([]byte, 4)
	binary.LittleEndian.PutUint32(bs, uint32(levelStart))
	crypto.WriteFullFile(out_path+"_Dec_"+strconv.Itoa(pid)+"_level", bs)
	crypto.WriteFullFile(out_path+"_Dec_"+strconv.Itoa(pid)+"_sizes", sizesDec)
	crypto.WriteFullFile(out_path+"_Dec_"+strconv.Itoa(pid), bytesDec)

	sizesRec, bytesRec := mpc.MarshalPolyMat(refSharesRecrypt)
	crypto.WriteFullFile(out_path+"_Rec_"+strconv.Itoa(pid)+"_sizes", sizesRec)
	crypto.WriteFullFile(out_path+"_Rec_"+strconv.Itoa(pid), bytesRec)

}

func CollectiveBootstrapClientReceive(pid int, vectorToBootstrapFile string, crpsPath string, server_polysPath string, out_path string) {
	cps := crypto.NewCryptoParamsFromDisk(false, pid, 1)
	parameters := cps.Params
	//skShard := cps.Sk.Value
	vectorToBootstrap, _ := crypto.LoadCipherMatrixFromFile(cps, vectorToBootstrapFile)
	cm := vectorToBootstrap

	// read saved level from send TODO maybe not needed
	//readLevelsBytes, err := crypto.LoadFullFile(levelPath + "_level")
	//level := int(binary.LittleEndian.Uint32(readLevelsBytes))
	// read both polys from server
	sizesPoly1Bytes, err := crypto.LoadFullFile(server_polysPath + "_1_sizes")
	if err != nil {
		log.Error("sizesPoly1Bytes ", err)
	}
	Poly1Bytes, err := crypto.LoadFullFile(server_polysPath + "_1")
	if err != nil {
		log.Error("Poly1Bytes ", err)
	}
	poly1 := mpc.UnmarshalPolyMat(sizesPoly1Bytes, Poly1Bytes)
	sizesPoly2Bytes, err := crypto.LoadFullFile(server_polysPath + "_2_sizes")
	if err != nil {
		log.Error("sizesPoly2Bytes ", err)
	}
	Poly2Bytes, err := crypto.LoadFullFile(server_polysPath + "_2")
	if err != nil {
		log.Error("Poly2Bytes ", err)
	}
	poly2 := mpc.UnmarshalPolyMat(sizesPoly2Bytes, Poly2Bytes)

	sizesCrps, err := crypto.LoadFullFile(crpsPath + "_" + strconv.Itoa(pid) + "_crpsSizes")
	if err != nil {
		log.Error("sizesCrps ", err)
	}
	bytesCrps, err := crypto.LoadFullFile(crpsPath + "_" + strconv.Itoa(pid) + "_crps")
	if err != nil {
		log.Error("bytesCrps ", err)
	}
	crps := mpc.UnmarshalPolyMat(sizesCrps, bytesCrps)

	// TODO change
	//token, _ := crypto.LoadFullFile("token_bootstrapping_1")
	//token2, _ := crypto.LoadFullFile("token_bootstrapping_2")
	token := make([]byte, 32)
	token2 := make([]byte, 32)
	refProtocol := dckks.NewRefreshProtocolDeterPRNGs(parameters, token, token2)

	for i := range cm {
		for j := range cm[i] {
			//no communication
			refProtocol.Decrypt(cm[i][j], &poly1[i][j])              // Masked decryption
			refProtocol.Recode(cm[i][j], parameters.Scale())         // Masked re-encoding
			refProtocol.Recrypt(cm[i][j], &crps[i][j], &poly2[i][j]) // Masked re-encryption

			// Fix discrepancy in number of moduli
			if len(cm[i][j].Value()[0].Coeffs) < len(cm[i][j].Value()[1].Coeffs) {
				poly := ring.NewPoly(len(cm[i][j].Value()[0].Coeffs[0]), len(cm[i][j].Value()[0].Coeffs))
				for pi := range poly.Coeffs {
					for pj := range poly.Coeffs[0] {
						poly.Coeffs[pi][pj] = cm[i][j].Value()[1].Coeffs[pi][pj]
					}
				}
				cm[i][j].Value()[1] = poly
			}
		}
	}
	crypto.SaveCipherMatrixToFile(cps, cm, out_path)
}

func CollectiveBootstrapServer(vectorsToBootstrap string, out_path string) {
	cps := crypto.NewCryptoParamsFromDisk(true, 1, 1)

	// TODO is this deterministic ? otherwise need to save RingQP
	dckksContext := dckks.NewContext(cps.Params)
	contextQ := dckksContext.RingQ
	fileIndex := 1
	var polysToBootstrap [][][]ring.Poly
	var polysToBootstrapMax [][][]ring.Poly
	var levels []int
	for {
		readsizesBytesDec, err := crypto.LoadFullFile(vectorsToBootstrap + "_Dec_" + strconv.Itoa(fileIndex) + "_sizes")
		if err != nil {
			break
		}
		readPolysBytesDec, err := crypto.LoadFullFile(vectorsToBootstrap + "_Dec_" + strconv.Itoa(fileIndex))
		readLevelsBytesDec, err := crypto.LoadFullFile(vectorsToBootstrap + "_Dec_" + strconv.Itoa(fileIndex) + "_level")
		levels = append(levels, int(binary.LittleEndian.Uint32(readLevelsBytesDec)))
		polyData := mpc.UnmarshalPolyMat(readsizesBytesDec, readPolysBytesDec)
		polysToBootstrap = append(polysToBootstrap, polyData)
		//TODO do better if changes smth
		readsizesBytesRec, err := crypto.LoadFullFile(vectorsToBootstrap + "_Rec_" + strconv.Itoa(fileIndex) + "_sizes")
		readPolysBytesRec, err := crypto.LoadFullFile(vectorsToBootstrap + "_Rec_" + strconv.Itoa(fileIndex))
		polyData2 := mpc.UnmarshalPolyMat(readsizesBytesRec, readPolysBytesRec)
		polysToBootstrapMax = append(polysToBootstrapMax, polyData2)

		log.LLvl1("SERVER ", fileIndex)
		fileIndex++
	}
	shareOut := make([][]ring.Poly, len(polysToBootstrap[0]))
	shareOutMaxLevel := make([][]ring.Poly, len(polysToBootstrapMax[0]))
	// Initialize
	for i := range shareOut {
		shareOut[i] = make([]ring.Poly, len(polysToBootstrap[0][i]))
		for j := range shareOut[i] {
			log.LLvl1("LEVEL ", levels[i])
			shareOut[i][j] = *contextQ.NewPolyLvl(levels[i])
		}
	}
	// Initialize
	for i := range shareOutMaxLevel {
		shareOutMaxLevel[i] = make([]ring.Poly, len(polysToBootstrapMax[0][i]))
		for j := range shareOutMaxLevel[i] {
			log.LLvl1("LEVEL ", levels[i])
			shareOutMaxLevel[i][j] = *contextQ.NewPolyLvl(cps.Params.MaxLevel())
		}
	}

	// aggregate both
	for p := range polysToBootstrap {
		for i := range shareOut {
			for j := range shareOut[i] {
				contextQ.AddLvl(len(polysToBootstrap[p][i][j].Coeffs)-1, &polysToBootstrap[p][i][j], &shareOut[i][j], &shareOut[i][j])
			}
		}
	}
	for p := range polysToBootstrapMax {
		for i := range shareOutMaxLevel {
			for j := range shareOutMaxLevel[i] {
				contextQ.AddLvl(len(polysToBootstrapMax[p][i][j].Coeffs)-1, &polysToBootstrapMax[p][i][j], &shareOutMaxLevel[i][j], &shareOutMaxLevel[i][j])
			}
		}
	}
	// save to file to send
	sizes, bytes := mpc.MarshalPolyMat(shareOut)
	crypto.WriteFullFile(out_path+"_1_sizes", sizes)
	crypto.WriteFullFile(out_path+"_1", bytes)
	sizesMax, bytesMax := mpc.MarshalPolyMat(shareOutMaxLevel)
	crypto.WriteFullFile(out_path+"_2_sizes", sizesMax)
	crypto.WriteFullFile(out_path+"_2", bytesMax)
}

// REF CollectiveDecryptMat
func CollectiveDecryptClientSend(pid int, vectorToDecryptFile string, out_path string) {
	cps := crypto.NewCryptoParamsFromDisk(false, pid, 1)
	parameters := cps.Params
	skShard := cps.Sk.Value
	vectorToBootstrap, _ := crypto.LoadCipherMatrixFromFile(cps, vectorToDecryptFile)
	tmp := vectorToBootstrap
	nr := len(tmp)
	nc := len(tmp[0])
	level := tmp[0][0].Level()

	zeroPoly := parameters.NewPolyQP()

	zeroPk := new(ckks.PublicKey)
	zeroPk.Value = [2]*ring.Poly{zeroPoly, zeroPoly}

	// save key for reuse
	token := make([]byte, 32)
	rand.Read(token)
	crypto.WriteFullFile("token_decryption", token)
	pcksProtocol := dckks.NewPCKSProtocolDeterPRNG(parameters, 6.36, token)

	decShare := make([][]dckks.PCKSShare, nr)
	for i := range decShare {
		decShare[i] = make([]dckks.PCKSShare, nc)
		for j := range decShare[i] {
			decShare[i][j] = pcksProtocol.AllocateShares(level)
			pcksProtocol.GenShare(skShard, zeroPk, tmp[i][j], decShare[i][j])
		}
	}

	// TODO add mask
	// decAgg
	for r := range decShare {
		for c := range decShare[r] {
			for i := range decShare[r][c] {
				decShareBytes, _ := decShare[r][c][i].MarshalBinary()
				crypto.WriteFullFile(out_path+"_"+strconv.Itoa(pid)+"_"+strconv.Itoa(r)+"_"+strconv.Itoa(c)+"_"+strconv.Itoa(i), decShareBytes)
			}
		}
	}
}

func CollectiveDecryptServer(vectorsToDecFile string, in_path string, out_path string) {
	cps := crypto.NewCryptoParamsFromDisk(true, 1, 1)

	parameters := cps.Params
	//skShard := cps.Sk.Value
	vectorsToDec, _ := crypto.LoadCipherMatrixFromFile(cps, vectorsToDecFile)
	tmp := vectorsToDec
	nr := len(tmp)
	nc := len(tmp[0])
	//level := tmp[0][0].Level()

	zeroPoly := parameters.NewPolyQP()

	zeroPk := new(ckks.PublicKey)
	zeroPk.Value = [2]*ring.Poly{zeroPoly, zeroPoly}

	// TODO save state for this
	//pcksProtocol := dckks.NewPCKSProtocol(parameters, 6.36)
	dckksContext := dckks.NewContext(cps.Params)

	// init
	decShare := make([][]dckks.PCKSShare, nr)
	for r := range decShare {
		decShare[r] = make([]dckks.PCKSShare, nc)
		for c := range decShare[r] {
			decShare[r][c] = zeroPk.Value //dckks.PCKSShare{new(ring.Poly), new(ring.Poly)}
		}
	}

	fileIndex := 1

	for {
		_, err := crypto.LoadFullFile(in_path + "_" + strconv.Itoa(fileIndex) + "_0_0_0")
		if err != nil {
			break
		}
		for r := range decShare {
			for c := range decShare[r] {
				for i := range decShare[r][c] {
					newPolyClient := new(ring.Poly)
					readPolyBytes, err := crypto.LoadFullFile(in_path + "_" + strconv.Itoa(fileIndex) + "_" + strconv.Itoa(r) + "_" + strconv.Itoa(c) + "_" + strconv.Itoa(i))
					log.LLvl1(err)
					newPolyClient.UnmarshalBinary(readPolyBytes)
					level := len(newPolyClient.Coeffs) - 1
					dckksContext.RingQ.AddLvl(level, newPolyClient, decShare[r][c][i], decShare[r][c][i])
				}
			}
		}
		fileIndex++
	}
	// save to file to send
	for r := range decShare {
		for c := range decShare[r] {
			for i := range decShare[r][c] {
				decShareBytes, _ := decShare[r][c][i].MarshalBinary()
				crypto.WriteFullFile(out_path+"_"+strconv.Itoa(r)+"_"+strconv.Itoa(c)+"_"+strconv.Itoa(i), decShareBytes)
			}
		}
	}
}

func CollectiveDecryptClientReceive(pid int, vectorToDecryptFile string, server_polysPath string, out_path string) {
	cps := crypto.NewCryptoParamsFromDisk(false, pid, 1)
	parameters := cps.Params
	//skShard := cps.Sk.Value
	vectorToDecrypt, _ := crypto.LoadCipherMatrixFromFile(cps, vectorToDecryptFile)
	cm := vectorToDecrypt
	nr := len(cm)
	nc := len(cm[0])
	log.LLvl1("AAAH", nr, nc)
	level := cm[0][0].Level()
	scale := cm[0][0].Scale()

	token, _ := crypto.LoadFullFile("token_decryption")
	pcksProtocol := dckks.NewPCKSProtocolDeterPRNG(parameters, 6.36, token)
	//dckksContext := dckks.NewContext(cps.Params)

	decShare := make([][]dckks.PCKSShare, nr)
	for r := range decShare {
		decShare[r] = make([]dckks.PCKSShare, nc)
		//for c := range decShare[r] {
		//	decShare[r][c] = pcksProtocol.AllocateShares(level)
		//}
	}

	for r := range decShare {
		for c := range decShare[r] {
			for i := range decShare[r][c] {
				newPolyServer := new(ring.Poly)
				readPolyBytes, err := crypto.LoadFullFile(server_polysPath + "_" + strconv.Itoa(r) + "_" + strconv.Itoa(c) + "_" + strconv.Itoa(i))
				log.LLvl1(err)
				newPolyServer.UnmarshalBinary(readPolyBytes)
				decShare[r][c][i] = newPolyServer
				//level := len(newPoly.Coeffs) - 1
				//dckksContext.RingQ.AddLvl(level, newPoly, decShare[r][c][i], decShare[r][c][i])
			}
		}
	}

	pm := make(crypto.PlainVector, 0)
	for i := range cm {
		for j := range cm[i] {
			ciphertextSwitched := ckks.NewCiphertext(parameters, 1, level, scale)
			pcksProtocol.KeySwitch(decShare[i][j], cm[i][j], ciphertextSwitched)
			pm = append(pm, ciphertextSwitched.Plaintext())
		}
	}
	log.LLvl1(len(pm), len(cm), len(cm[0]))
	pmDecoded := crypto.DecodeFloatVector(cps, pm)

	crypto.SaveFloatVectorToFile(out_path, pmDecoded)
}
