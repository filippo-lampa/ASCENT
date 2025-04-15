const { expect } = require('chai'); 
const { ethers, upgrades } = require('hardhat'); 

describe('Forwarder', function () { 
    let forwarder; 
    let owner; 
    let admin; 
    let eurftoken; 

    beforeEach(async function () { 
        [owner, admin] = await ethers.getSigners(); 
        const EURFToken = await ethers.getContractFactory('EURFToken'); 
        eurftoken = await upgrades.deployProxy(EURFToken, [], { kind: 'uups', initializer: 'initialize' }); 
        const Forwarder = await ethers.getContractFactory('Forwarder'); 
        forwarder = await upgrades.deployProxy(Forwarder, [eurftoken.target], { initializer: 'initialize' }); 
    }); 

    describe('registerRequestType', function () { 
        it('should revert when typeName contains invalid character ")"', async function () { 
            const invalidTypeName = 'invalidType)'; 
            const typeSuffix = 'suffix'; 
            await expect(forwarder.connect(owner).registerRequestType(invalidTypeName, typeSuffix)) 
                .to.be.revertedWith("NGEUR Forwarder: invalid typename"); 
        }); 
    }); 
});