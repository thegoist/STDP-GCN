  def save_as_h5(self, file_path, name):
        
        f = h5py.File(file_path + name + "0.h5", "w")

        image_list = {
            "256":[],
            "256_image":[],
            "256_index":[],
            "256_label":[],
            "2048":[],
            "2048_image":[],
            "2048_index":[],
            "2048_label":[],
        }
        
        fits_index = 1
        total = len(self.source_file)
        start_time = datetime.datetime.now()
        for fits_object in self.source_file:
            fits_path = fits_object['image_path']
            hdulist = pyfits.open(fits_path)
            hdu1 = hdulist[1]
            fits_data = hdu1.data['data'].reshape(-1,2048)
            print(fits_index,":",fits_path)
            #logger.debug(fits_index,":",fits_path)
            for crop_image in fits_object["256"]:
                

                data, frb_index = self.analyze_data( [ crop_image["index"], fits_path], need_index = True, image = fits_data)

                image_list["256"].append(  copy.deepcopy( data ) )
                image_list["256_index"].append( copy.deepcopy( frb_index ) )
                image_list["256_image"].append(fits_index)
                image_list["256_label"].append(crop_image["label"])
                
            
            for crop_image in fits_object["2048"]:
                data, frb_index = self.analyze_data( [ crop_image["index"], fits_path],  need_index = True, image = fits_data)

                image_list["2048"].append(  copy.deepcopy( data ) )
                image_list["2048_index"].append( copy.deepcopy( frb_index ) )
                image_list["2048_image"].append(fits_index)
                image_list["2048_label"].append(crop_image["label"])

            hdulist.close()
            if( fits_index%1 == 0):
                end_time = datetime.datetime.now()
                print("label #{} {}/{} EAT:{} minute".format(int(fits_index/total*100),fits_index,total, int( (end_time - start_time).seconds/60*100 )/100))
                start_time = def save_as_h5(self, file_path, name):
        
        f = h5py.File(file_path + name + "0.h5", "w")

        image_list = {
            "256":[],
            "256_image":[],
            "256_index":[],
            "256_label":[],
            "2048":[],
            "2048_image":[],
            "2048_index":[],
            "2048_label":[],
        }
        
        fits_index = 1
        total = len(self.source_file)
        start_time = datetime.datetime.now()
        for fits_object in self.source_file:
            fits_path = fits_object['image_path']
            hdulist = pyfits.open(fits_path)
            hdu1 = hdulist[1]
            fits_data = hdu1.data['data'].reshape(-1,2048)
            print(fits_index,":",fits_path)
            #logger.debug(fits_index,":",fits_path)
            for crop_image in fits_object["256"]:
                

                data, frb_index = self.analyze_data( [ crop_image["index"], fits_path], need_index = True, image = fits_data)

                image_list["256"].append(  copy.deepcopy( data ) )
                image_list["256_index"].append( copy.deepcopy( frb_index ) )
                image_list["256_image"].append(fits_index)
                image_list["256_label"].append(crop_image["label"])
                
            
            for crop_image in fits_object["2048"]:
                data, frb_index = self.analyze_data( [ crop_image["index"], fits_path],  need_index = True, image = fits_data)

                image_list["2048"].append(  copy.deepcopy( data ) )
                image_list["2048_index"].append( copy.deepcopy( frb_index ) )
                image_list["2048_image"].append(fits_index)
                image_list["2048_label"].append(crop_image["label"])

            hdulist.close()
            if( fits_index%1 == 0):
                end_time = datetime.datetime.now()
                print("label #{} {}/{} EAT:{} minute".format(int(fits_index/total*100),fits_index,total, int( (end_time - start_time).seconds/60*100 )/100))
                start_time = if( fits_index%1 == 0):
                end_time = datetime.datetime.now()
                print("label #{} {}/{} EAT:{} minute".format(int(fits_index/total*100),fits_index,total, int( (end_time - start_time).seconds/60*100 )/100))
                start_time = end_time

                group_256 = f.create_group("256")
                group_256.create_dataset("image", data = image_list["256"])
                group_256.create_dataset("label", data = image_list["256_label"])
                group_256.create_dataset("coord", data = image_list["256_index"])
                group_256.create_dataset("index", data = image_list["256_image"])

                group_2048 = f.create_group("2048")
                group_2048.create_dataset("image", data = image_list["2048"])
                group_2048.create_dataset("label", data = image_list["2048_label"])
                group_2048.create_dataset("coord", data = image_list["2048_index"])
                group_2048.create_dataset("index", data = image_list["2048_image"])
                f.close()
                del image_list
                image_list = {
                    "256":[],
                    "256_image":[],
                    "256_index":[],
                    "256_label":[],
                    "2048":[],
                    "2048_image":[],
                    "2048_index":[],
                    "2048_label":[],
                }
                f = h5py.File(file_path + name + str(fits_index)+ ".h5", "w")

            fits_index += 1
        f.close()